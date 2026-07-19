// =============================================================================
// analyzer.cu - stage 2c structural analysis implementation
// =============================================================================
//
// All statistical analyses run on HOST (data sizes are small: 10k neurons,
// 256 byte values, 1M synapses). GPU is used only for forward propagation
// to collect activation data.
//
// No external dependencies (no Eigen, no LAPACK). PCA uses power iteration,
// K-means uses Lloyd's algorithm, all implemented from scratch.
// =============================================================================

#include "analyzer.cuh"
#include "config.h"
#include "../include/config.h"
#include "../include/types.h"
#include "competition.cuh"            // P2: k-WTA (must match training)

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <cuda_runtime.h>

// =============================================================================
// 1. Collect activation data (GPU forward + host accumulation)
// =============================================================================

ActivationData collect_activation_data(
    SNNNetwork& network,
    TextStream& stream,
    long n_steps
) {
    ActivationData data;
    data.neuron_spike_count.assign(N_TOTAL_NEURONS, 0);
    data.byte_activation.assign(256 * N_TOTAL_NEURONS, 0);
    data.byte_count.assign(256, 0);
    data.step_total_spikes.reserve(n_steps);
    data.total_steps = n_steps;
    data.total_spikes = 0;

    // 方案 A: 输入缓冲区长度 = N_TOTAL_NEURONS（sensory 神经元分散在每柱内）
    float* h_input = new float[N_TOTAL_NEURONS];
    bool*  h_spikes = new bool[N_TOTAL_NEURONS];

    std::cout << "[Analyzer] Collecting activation data over " << n_steps
              << " steps..." << std::endl;

    for (long t = 0; t < n_steps; t++) {
        unsigned char b = stream.next_byte();
        stream.build_sensory_input(b, h_input);
        network.step(h_input, (int)t);

        // 纯 SNN 实验：移除 k-WTA 外部竞争（与训练保持一致）
        // 原先: apply_kwta_competition(network, STAGE2_KWTA_K);

        // Pull spikes to host
        CUDA_CHECK(cudaMemcpy(h_spikes, network.get_d_spikes(),
                              N_TOTAL_NEURONS * sizeof(bool),
                              cudaMemcpyDeviceToHost));

        long step_spikes = 0;
        for (int n = 0; n < N_TOTAL_NEURONS; n++) {
            if (h_spikes[n]) {
                data.neuron_spike_count[n]++;
                data.byte_activation[(int)b * N_TOTAL_NEURONS + n]++;
                step_spikes++;
            }
        }
        data.byte_count[b]++;
        data.step_total_spikes.push_back(step_spikes);
        data.total_spikes += step_spikes;

        if (t > 0 && t % 1000 == 0) {
            std::cout << "  step " << t << "/" << n_steps
                      << " total_spikes=" << data.total_spikes
                      << "\r" << std::flush;
        }
    }
    std::cout << std::endl;

    delete[] h_input;
    delete[] h_spikes;

    std::cout << "[Analyzer] Done. total_spikes=" << data.total_spikes
              << " (avg " << (double)data.total_spikes / n_steps
              << " spikes/step)" << std::endl;

    return data;
}

// =============================================================================
// 2. Power-law analysis
// =============================================================================
// Fit P(X > x) ~ x^(-alpha) using log-log linear regression on CCDF.
// =============================================================================

PowerLawResult analyze_power_law(const ActivationData& data) {
    PowerLawResult result;
    result.alpha = 0;
    result.r_squared = 0;
    result.min_count = 0;
    result.max_count = 0;
    result.mean_count = 0;
    result.median_count = 0;
    result.n_active = 0;
    result.n_silent = 0;
    result.is_power_law = false;

    // Find active neurons (>=1 spike)
    std::vector<long> counts;
    counts.reserve(N_TOTAL_NEURONS);
    for (int n = 0; n < N_TOTAL_NEURONS; n++) {
        if (data.neuron_spike_count[n] > 0) {
            counts.push_back(data.neuron_spike_count[n]);
        }
    }
    result.n_active = counts.size();
    result.n_silent = N_TOTAL_NEURONS - result.n_active;

    if (counts.empty()) {
        result.alpha = 0;
        result.r_squared = 0;
        result.is_power_law = false;
        return result;
    }

    // Sort descending
    std::sort(counts.begin(), counts.end(), std::greater<long>());

    result.min_count = counts.back();
    result.max_count = counts.front();

    double sum = std::accumulate(counts.begin(), counts.end(), 0.0);
    result.mean_count = sum / counts.size();
    result.median_count = counts[counts.size() / 2];

    // Build CCDF: P(X >= x) for each unique x
    // Use rank-based: P(X >= counts[i]) = (i+1) / N
    // Then fit log(rank) vs log(counts[i])
    std::vector<double> log_x, log_ccdf;
    long prev = -1;
    for (size_t i = 0; i < counts.size(); i++) {
        if (counts[i] == prev) continue;
        prev = counts[i];
        double x = (double)counts[i];
        double ccdf = (double)(i + 1) / counts.size();
        if (x > 0 && ccdf > 0) {
            log_x.push_back(std::log(x));
            log_ccdf.push_back(std::log(ccdf));
        }
    }

    if (log_x.size() < 2) {
        result.alpha = 0;
        result.r_squared = 0;
        result.is_power_law = false;
        return result;
    }

    // Linear regression: log(ccdf) = a - alpha * log(x)
    // => y = a + b*x where b = -alpha
    double n = log_x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (size_t i = 0; i < log_x.size(); i++) {
        sum_x += log_x[i];
        sum_y += log_ccdf[i];
        sum_xy += log_x[i] * log_ccdf[i];
        sum_x2 += log_x[i] * log_x[i];
    }
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double a = mean_y - b * mean_x;

    result.alpha = -b;

    // R^2
    double ss_tot = 0, ss_res = 0;
    for (size_t i = 0; i < log_x.size(); i++) {
        double y_pred = a + b * log_x[i];
        ss_res += (log_ccdf[i] - y_pred) * (log_ccdf[i] - y_pred);
        ss_tot += (log_ccdf[i] - mean_y) * (log_ccdf[i] - mean_y);
    }
    result.r_squared = (ss_tot > 0) ? (1.0 - ss_res / ss_tot) : 0.0;

    result.is_power_law = (result.alpha >= 1.5 && result.alpha <= 3.0
                           && result.r_squared > 0.9);

    return result;
}

// =============================================================================
// 3. Chi-square test of byte-neuron independence
// =============================================================================
// For each neuron n with >= min_spikes total spikes:
//   H0: neuron n's firing is independent of input byte
//   obs[b] = byte_activation[b][n]
//   exp[b] = byte_count[b] * neuron_spike_count[n] / total_steps
//   chi2 = sum_b (obs[b] - exp[b])^2 / exp[b]
//   df = (n_bytes_seen - 1)
// =============================================================================

ChiSquareResult analyze_chi_square(const ActivationData& data) {
    ChiSquareResult result;
    result.n_total = N_TOTAL_NEURONS;
    result.df = 0;
    result.n_significant = 0;
    result.mean_chi2 = 0;
    result.max_chi2 = 0;

    // Count distinct bytes seen
    int n_bytes_seen = 0;
    for (int b = 0; b < 256; b++) {
        if (data.byte_count[b] > 0) n_bytes_seen++;
    }
    result.df = n_bytes_seen - 1;
    if (result.df < 1) return result;

    // Chi-square critical value for p=0.01, df=255 is ~310.457
    // For general df, use approximation: chi2_crit ~= df + 2.33*sqrt(2*df) + ... 
    // For p=0.01: z=2.33, chi2_crit ~= df + z*sqrt(2*df) + (z^2-1)/3
    double z = 2.33;  // p=0.01 one-tailed
    double chi2_crit = result.df + z * std::sqrt(2.0 * result.df)
                       + (z * z - 1.0) / 3.0;

    long min_spikes = 10;  // skip neurons with too few spikes for reliable test
    double chi2_sum = 0;
    long n_tested = 0;

    for (int n = 0; n < N_TOTAL_NEURONS; n++) {
        if (data.neuron_spike_count[n] < min_spikes) continue;

        double chi2 = 0;
        double total_n = data.neuron_spike_count[n];
        for (int b = 0; b < 256; b++) {
            if (data.byte_count[b] == 0) continue;
            double exp = (double)data.byte_count[b] * total_n / data.total_steps;
            if (exp < 5) continue;  // chi-square requires expected >= 5
            double obs = data.byte_activation[b * N_TOTAL_NEURONS + n];
            double diff = obs - exp;
            chi2 += diff * diff / exp;
        }

        chi2_sum += chi2;
        n_tested++;
        if (chi2 > chi2_crit) result.n_significant++;
        if (chi2 > result.max_chi2) result.max_chi2 = chi2;
    }

    result.n_total = n_tested;
    result.mean_chi2 = (n_tested > 0) ? chi2_sum / n_tested : 0;
    result.fraction_significant = (n_tested > 0)
        ? (double)result.n_significant / n_tested : 0;
    result.is_significant = (result.fraction_significant > 0.5);

    return result;
}

// =============================================================================
// 4. PCA via power iteration (no external libs)
// =============================================================================
// Input: 256-dim vectors per neuron (the byte_activation rows)
// We compute top-2 principal components of the 256x256 covariance matrix.
// =============================================================================

static void pca_2d(
    const std::vector<long>& byte_activation,  // [256 * N]
    int n_neurons,
    std::vector<double>& coords,    // out: [n_neurons * 2]
    double& variance_explained
) {
    // Step 1: compute mean byte-response per neuron
    std::vector<double> mean_resp(n_neurons, 0);
    for (int b = 0; b < 256; b++) {
        for (int n = 0; n < n_neurons; n++) {
            mean_resp[n] += byte_activation[b * n_neurons + n];
        }
    }
    for (int n = 0; n < n_neurons; n++) mean_resp[n] /= 256.0;

    // Step 2: build 256x256 covariance matrix
    // cov[b1][b2] = sum_n (X[n][b1] - mean_b1) * (X[n][b2] - mean_b2) / N
    // But we want PCA of neurons, so we compute neuron x neuron covariance
    // Actually: we want to project neurons into 2D based on their byte-response
    // patterns. So each neuron is a 256-dim point, and we do PCA on the
    // 256x256 covariance (scatter matrix) of these points.
    //
    // Simplification: compute 256x256 scatter matrix S[b1][b2]
    std::vector<double> scatter(256 * 256, 0);
    for (int b1 = 0; b1 < 256; b1++) {
        for (int b2 = b1; b2 < 256; b2++) {
            double s = 0;
            for (int n = 0; n < n_neurons; n++) {
                double x1 = byte_activation[b1 * n_neurons + n] - mean_resp[n];
                double x2 = byte_activation[b2 * n_neurons + n] - mean_resp[n];
                s += x1 * x2;
            }
            scatter[b1 * 256 + b2] = s;
            scatter[b2 * 256 + b1] = s;
        }
    }

    // Step 3: power iteration to find top-2 eigenvectors
    std::vector<double> v1(256, 1.0 / std::sqrt(256));
    std::vector<double> v2(256, 0);
    double eigenvalue1 = 0, eigenvalue2 = 0;

    // First eigenvector
    for (int iter = 0; iter < 100; iter++) {
        std::vector<double> Av(256, 0);
        for (int b1 = 0; b1 < 256; b1++) {
            for (int b2 = 0; b2 < 256; b2++) {
                Av[b1] += scatter[b1 * 256 + b2] * v1[b2];
            }
        }
        double norm = 0;
        for (int b = 0; b < 256; b++) norm += Av[b] * Av[b];
        norm = std::sqrt(norm);
        if (norm < 1e-12) break;
        for (int b = 0; b < 256; b++) v1[b] = Av[b] / norm;
        eigenvalue1 = norm;
    }

    // Deflate: S' = S - eigenvalue1 * v1 * v1^T
    std::vector<double> scatter2 = scatter;
    for (int b1 = 0; b1 < 256; b1++) {
        for (int b2 = 0; b2 < 256; b2++) {
            scatter2[b1 * 256 + b2] -= eigenvalue1 * v1[b1] * v1[b2];
        }
    }

    // Second eigenvector
    for (int b = 0; b < 256; b++) v2[b] = (b == 0) ? 1.0 : 0.0;
    for (int iter = 0; iter < 100; iter++) {
        std::vector<double> Av(256, 0);
        for (int b1 = 0; b1 < 256; b1++) {
            for (int b2 = 0; b2 < 256; b2++) {
                Av[b1] += scatter2[b1 * 256 + b2] * v2[b2];
            }
        }
        double norm = 0;
        for (int b = 0; b < 256; b++) norm += Av[b] * Av[b];
        norm = std::sqrt(norm);
        if (norm < 1e-12) break;
        for (int b = 0; b < 256; b++) v2[b] = Av[b] / norm;
        eigenvalue2 = norm;
    }

    // Step 4: project neurons onto PC1 and PC2
    coords.assign(n_neurons * 2, 0);
    for (int n = 0; n < n_neurons; n++) {
        double pc1 = 0, pc2 = 0;
        for (int b = 0; b < 256; b++) {
            double x = byte_activation[b * n_neurons + n] - mean_resp[n];
            pc1 += v1[b] * x;
            pc2 += v2[b] * x;
        }
        coords[n * 2] = pc1;
        coords[n * 2 + 1] = pc2;
    }

    // Variance explained by top 2 PCs
    double total_var = 0;
    for (int b = 0; b < 256; b++) total_var += scatter[b * 256 + b];
    variance_explained = (total_var > 0)
        ? (eigenvalue1 + eigenvalue2) / total_var : 0;
}

// =============================================================================
// 5. K-means clustering (Lloyd's algorithm)
// =============================================================================

static void kmeans(
    const std::vector<double>& points,  // [n * 2]
    int n,
    int k,
    std::vector<int>& assignments,
    std::vector<double>& centroids,
    int max_iter = 100
) {
    std::mt19937 rng(42);
    assignments.assign(n, 0);
    centroids.assign(k * 2, 0);

    // K-means++ initialization
    std::uniform_int_distribution<int> uniform(0, n - 1);
    int first = uniform(rng);
    centroids[0] = points[first * 2];
    centroids[1] = points[first * 2 + 1];

    for (int c = 1; c < k; c++) {
        std::vector<double> dists(n, 1e18);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < c; j++) {
                double dx = points[i * 2] - centroids[j * 2];
                double dy = points[i * 2 + 1] - centroids[j * 2 + 1];
                double d = dx * dx + dy * dy;
                if (d < dists[i]) dists[i] = d;
            }
        }
        double sum = std::accumulate(dists.begin(), dists.end(), 0.0);
        if (sum <= 0) {
            // All points identical; just pick random
            int idx = uniform(rng);
            centroids[c * 2] = points[idx * 2];
            centroids[c * 2 + 1] = points[idx * 2 + 1];
            continue;
        }
        std::uniform_real_distribution<double> uniform01(0, sum);
        double r = uniform01(rng);
        double cum = 0;
        int chosen = n - 1;
        for (int i = 0; i < n; i++) {
            cum += dists[i];
            if (cum >= r) { chosen = i; break; }
        }
        centroids[c * 2] = points[chosen * 2];
        centroids[c * 2 + 1] = points[chosen * 2 + 1];
    }

    // Lloyd iterations
    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;
        // Assign
        for (int i = 0; i < n; i++) {
            double best_d = 1e18;
            int best_c = 0;
            for (int c = 0; c < k; c++) {
                double dx = points[i * 2] - centroids[c * 2];
                double dy = points[i * 2 + 1] - centroids[c * 2 + 1];
                double d = dx * dx + dy * dy;
                if (d < best_d) { best_d = d; best_c = c; }
            }
            if (assignments[i] != best_c) {
                assignments[i] = best_c;
                changed = true;
            }
        }
        // Update
        std::vector<double> new_centroids(k * 2, 0);
        std::vector<int> counts(k, 0);
        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            new_centroids[c * 2] += points[i * 2];
            new_centroids[c * 2 + 1] += points[i * 2 + 1];
            counts[c]++;
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                centroids[c * 2] = new_centroids[c * 2] / counts[c];
                centroids[c * 2 + 1] = new_centroids[c * 2 + 1] / counts[c];
            }
        }
        if (!changed && iter > 0) break;
    }
}

// =============================================================================
// 6. Silhouette score
// =============================================================================
// For each point i:
//   a(i) = mean distance to other points in same cluster
//   b(i) = min over other clusters of mean distance to points in that cluster
//   s(i) = (b - a) / max(a, b)
// silhouette = mean of s(i)
// =============================================================================

static double silhouette_score(
    const std::vector<double>& points,
    const std::vector<int>& assignments,
    int n,
    int k
) {
    // Precompute cluster memberships
    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < n; i++) clusters[assignments[i]].push_back(i);

    double total_s = 0;
    int valid = 0;
    for (int i = 0; i < n; i++) {
        int ci = assignments[i];
        if (clusters[ci].size() <= 1) continue;  // singleton: s=0 by convention

        // a(i): mean dist to same cluster
        double a = 0;
        for (int j : clusters[ci]) {
            if (j == i) continue;
            double dx = points[i * 2] - points[j * 2];
            double dy = points[i * 2 + 1] - points[j * 2 + 1];
            a += std::sqrt(dx * dx + dy * dy);
        }
        a /= (clusters[ci].size() - 1);

        // b(i): min mean dist to other clusters
        double b = 1e18;
        for (int c = 0; c < k; c++) {
            if (c == ci || clusters[c].empty()) continue;
            double dist = 0;
            for (int j : clusters[c]) {
                double dx = points[i * 2] - points[j * 2];
                double dy = points[i * 2 + 1] - points[j * 2 + 1];
                dist += std::sqrt(dx * dx + dy * dy);
            }
            dist /= clusters[c].size();
            if (dist < b) b = dist;
        }

        if (b < 1e18 && (a > 0 || b > 0)) {
            double s = (b - a) / std::max(a, b);
            total_s += s;
            valid++;
        }
    }
    return (valid > 0) ? total_s / valid : 0;
}

// =============================================================================
// 7. Cluster analysis: PCA + K-means + silhouette (test multiple K)
// =============================================================================

ClusterResult analyze_clusters(const ActivationData& data) {
    ClusterResult best;
    best.silhouette = -1;
    best.k = 0;
    best.variance_explained = 0;
    best.clusters_emerged = false;

    // PCA
    std::vector<double> coords;
    double var_exp;
    pca_2d(data.byte_activation, N_TOTAL_NEURONS, coords, var_exp);

    // Test K = 5, 8, 10, 12, 15, 20
    std::vector<int> ks = {5, 8, 10, 12, 15, 20};
    for (int k : ks) {
        std::vector<int> assignments;
        std::vector<double> centroids;
        kmeans(coords, N_TOTAL_NEURONS, k, assignments, centroids);
        double s = silhouette_score(coords, assignments, N_TOTAL_NEURONS, k);

        std::cout << "  K=" << k << " silhouette=" << s << std::endl;

        if (s > best.silhouette) {
            best.k = k;
            best.silhouette = s;
            best.assignments = assignments;
            best.centroids = centroids;
            best.pca_coords = coords;
            best.variance_explained = var_exp;
        }
    }

    best.clusters_emerged = (best.silhouette > 0.3
                             && best.k >= 5 && best.k <= 20);
    return best;
}

// =============================================================================
// 8. Weight distribution analysis
// =============================================================================

WeightStats analyze_weight_distribution(SNNNetwork& network) {
    WeightStats ws;
    ws.histogram.assign(20, 0);

    float* h_w = new float[N_TOTAL_SYNAPSES];
    CUDA_CHECK(cudaMemcpy(h_w, network.get_d_weights(),
                          N_TOTAL_SYNAPSES * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double sum = 0, sum_sq = 0;
    ws.min = h_w[0];
    ws.max = h_w[0];
    long n_pos = 0, n_zero = 0, n_sat = 0;
    double W_MAX = 1.0;  // from config.h STDP_W_MAX

    for (int i = 0; i < N_TOTAL_SYNAPSES; i++) {
        double w = h_w[i];
        sum += w;
        sum_sq += w * w;
        if (w < ws.min) ws.min = w;
        if (w > ws.max) ws.max = w;
        if (w > 0) n_pos++;
        if (std::abs(w) < 1e-6) n_zero++;
        if (std::abs(w) > 0.9 * W_MAX) n_sat++;

        // Histogram: 20 bins from -W_MAX to +W_MAX
        int bin = (int)((w + W_MAX) / (2 * W_MAX) * 20);
        if (bin < 0) bin = 0;
        if (bin >= 20) bin = 19;
        ws.histogram[bin]++;
    }

    ws.mean = sum / N_TOTAL_SYNAPSES;
    double var = sum_sq / N_TOTAL_SYNAPSES - ws.mean * ws.mean;
    ws.std_dev = (var > 0) ? std::sqrt(var) : 0;
    ws.fraction_positive = (double)n_pos / N_TOTAL_SYNAPSES;
    ws.fraction_zero = (double)n_zero / N_TOTAL_SYNAPSES;
    ws.fraction_saturated = (double)n_sat / N_TOTAL_SYNAPSES;

    delete[] h_w;
    return ws;
}

// =============================================================================
// 9. Report printing
// =============================================================================

void print_analysis_report(
    const std::string& label,
    const ActivationData& act,
    const PowerLawResult& pl,
    const ChiSquareResult& chi,
    const ClusterResult& clu,
    const WeightStats& ws
) {
    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Analysis Report: " << label << std::endl;
    std::cout << "============================================================" << std::endl;

    std::cout << std::endl << "--- Activation Summary ---" << std::endl;
    std::cout << "  total_steps      = " << act.total_steps << std::endl;
    std::cout << "  total_spikes     = " << act.total_spikes << std::endl;
    std::cout << "  avg spikes/step  = " << (double)act.total_spikes / act.total_steps << std::endl;
    std::cout << "  active neurons   = " << pl.n_active << " / " << N_TOTAL_NEURONS
              << " (" << (double)pl.n_active / N_TOTAL_NEURONS * 100 << "%)" << std::endl;
    std::cout << "  silent neurons   = " << pl.n_silent << std::endl;

    std::cout << std::endl << "--- Power-law Analysis ---" << std::endl;
    std::cout << "  alpha (exponent) = " << pl.alpha << std::endl;
    std::cout << "  R^2              = " << pl.r_squared << std::endl;
    std::cout << "  mean spike count = " << pl.mean_count << std::endl;
    std::cout << "  median count     = " << pl.median_count << std::endl;
    std::cout << "  min/max count    = " << pl.min_count << " / " << pl.max_count << std::endl;
    std::cout << "  verdict          = "
              << (pl.is_power_law ? "POWER-LAW (alpha in [1.5,3.0], R^2>0.9)"
                                  : "NOT power-law")
              << std::endl;

    std::cout << std::endl << "--- Chi-square Test (byte-neuron independence) ---" << std::endl;
    std::cout << "  neurons tested   = " << chi.n_total << std::endl;
    std::cout << "  degrees of free  = " << chi.df << std::endl;
    std::cout << "  mean chi2        = " << chi.mean_chi2 << std::endl;
    std::cout << "  max chi2         = " << chi.max_chi2 << std::endl;
    std::cout << "  significant      = " << chi.n_significant << " / " << chi.n_total
              << " (" << chi.fraction_significant * 100 << "%)" << std::endl;
    std::cout << "  verdict          = "
              << (chi.is_significant ? "SIGNIFICANT (>50% neurons discriminate bytes)"
                                      : "not significant")
              << std::endl;

    std::cout << std::endl << "--- PCA + K-means Clustering ---" << std::endl;
    std::cout << "  variance explained (top 2 PCs) = " << clu.variance_explained * 100 << "%" << std::endl;
    std::cout << "  best K          = " << clu.k << std::endl;
    std::cout << "  silhouette      = " << clu.silhouette << std::endl;
    std::cout << "  verdict         = "
              << (clu.clusters_emerged ? "CLUSTERS EMERGED (silhouette>0.3, 5<=K<=20)"
                                        : "no clear clusters")
              << std::endl;

    // Cluster sizes
    if (!clu.assignments.empty()) {
        std::vector<int> sizes(clu.k, 0);
        for (int a : clu.assignments) sizes[a]++;
        std::cout << "  cluster sizes   = ";
        for (int c = 0; c < clu.k; c++) {
            std::cout << sizes[c] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "--- Weight Distribution ---" << std::endl;
    std::cout << "  mean            = " << ws.mean << std::endl;
    std::cout << "  std_dev         = " << ws.std_dev << std::endl;
    std::cout << "  min/max         = " << ws.min << " / " << ws.max << std::endl;
    std::cout << "  fraction pos    = " << ws.fraction_positive * 100 << "% (excitatory)" << std::endl;
    std::cout << "  fraction zero   = " << ws.fraction_zero * 100 << "%" << std::endl;
    std::cout << "  fraction sat    = " << ws.fraction_saturated * 100 << "% (|w|>0.9*W_MAX)" << std::endl;
    std::cout << "  histogram (-1..+1, 20 bins):" << std::endl;
    for (int i = 0; i < 20; i++) {
        double lo = -1.0 + i * 0.1;
        double hi = lo + 0.1;
        std::cout << "    [" << lo << ", " << hi << "): " << ws.histogram[i] << std::endl;
    }

    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
}

// =============================================================================
// 10. Save cluster CSV for plotting
// =============================================================================

void save_cluster_csv(
    const std::string& path,
    const ClusterResult& clu
) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[Analyzer] Cannot open " << path << " for write" << std::endl;
        return;
    }
    ofs << "neuron_id,pc1,pc2,cluster,region\n";
    for (int n = 0; n < N_TOTAL_NEURONS; n++) {
        const char* region = "MOTOR";
        if (n < N_SENSORY_NEURONS) region = "SENSORY";
        else if (n < N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS) region = "ASSOCIATION";
        ofs << n << ","
            << clu.pca_coords[n * 2] << ","
            << clu.pca_coords[n * 2 + 1] << ","
            << clu.assignments[n] << ","
            << region << "\n";
    }
    std::cout << "[Analyzer] Saved " << N_TOTAL_NEURONS
              << " points to " << path << std::endl;
}
