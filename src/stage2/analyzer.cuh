#ifndef SNN_STAGE2_ANALYZER_CUH
#define SNN_STAGE2_ANALYZER_CUH

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../include/types.h"
#include "../include/network.h"
#include "text_stream.cuh"

// =============================================================================
// Stage 2c structural analysis tools
// =============================================================================
//
// All analysis runs on HOST (data is small: 10k neurons, 256 byte values).
// GPU is used only for forward propagation to collect activation data.
//
// Four analyses per v3 plan:
//   1. Power-law firing distribution (does firing follow Pareto?)
//   2. Chi-square byte-neuron independence test (do neurons discriminate?)
//   3. PCA + K-means clustering (do semantic clusters emerge?)
//   4. Silhouette score (cluster quality)
//
// Plus baseline B1 comparison (random weights, same topology, same input).
// =============================================================================

// -----------------------------------------------------------------------------
// Activation data collected from a test run
// -----------------------------------------------------------------------------
struct ActivationData {
    // Per-neuron total spike count [N_TOTAL_NEURONS]
    std::vector<long> neuron_spike_count;

    // Byte-neuron activation matrix [256 * N_TOTAL_NEURONS]
    // byte_activation[b * N + n] = times neuron n fired when byte b was input
    std::vector<long> byte_activation;

    // Per-byte occurrence count [256]
    std::vector<long> byte_count;

    // Per-step total spike count (for time-series analysis)
    std::vector<long> step_total_spikes;

    long total_steps;       // number of steps run
    long total_spikes;      // sum of all neuron_spike_count
};

// -----------------------------------------------------------------------------
// Power-law analysis result
// -----------------------------------------------------------------------------
struct PowerLawResult {
    double alpha;           // fitted exponent: P(X>x) ~ x^(-alpha)
    double r_squared;       // goodness of fit
    long   min_count;       // minimum spike count in data
    long   max_count;
    double mean_count;
    double median_count;
    long   n_active;        // neurons with >= 1 spike
    long   n_silent;        // neurons with 0 spikes
    bool   is_power_law;    // true if alpha in [1.5, 3.0] and R^2 > 0.9
};

// -----------------------------------------------------------------------------
// Chi-square test result
// -----------------------------------------------------------------------------
struct ChiSquareResult {
    long   n_significant;       // neurons with p < 0.01 (reject independence)
    long   n_total;             // total neurons tested
    double fraction_significant; // n_significant / n_total
    double mean_chi2;           // mean chi-square statistic
    double max_chi2;            // max chi-square statistic
    long   df;                  // degrees of freedom (255)
    bool   is_significant;      // true if >50% neurons significant
};

// -----------------------------------------------------------------------------
// PCA + K-means result
// -----------------------------------------------------------------------------
struct ClusterResult {
    int    k;                          // number of clusters
    std::vector<int> assignments;      // cluster id per neuron [N]
    std::vector<double> centroids;     // [k * 2] (2D PCA space)
    std::vector<double> pca_coords;    // [N * 2] 2D PCA projection
    double silhouette;                 // silhouette score (-1..1)
    double variance_explained;         // % variance by first 2 PCs
    bool   clusters_emerged;           // true if silhouette > 0.3 and 5<=k<=20
};

// -----------------------------------------------------------------------------
// Weight distribution stats
// -----------------------------------------------------------------------------
struct WeightStats {
    double mean;
    double std_dev;
    double min;
    double max;
    double fraction_positive;  // excitatory fraction
    double fraction_zero;      // silent synapses
    double fraction_saturated; // |w| > 0.9 * W_MAX
    std::vector<long> histogram;  // 20 bins from -W_MAX to +W_MAX
};

// =============================================================================
// Public API
// =============================================================================

// Run test sequence and collect activation data.
// Uses first `n_steps` bytes of the text stream.
ActivationData collect_activation_data(
    SNNNetwork& network,
    TextStream& stream,
    long n_steps
);

// Analysis 1: Power-law fit of firing distribution
PowerLawResult analyze_power_law(const ActivationData& data);

// Analysis 2: Chi-square test of byte-neuron independence
ChiSquareResult analyze_chi_square(const ActivationData& data);

// Analysis 3+4: PCA + K-means clustering + silhouette
// Tests K = 5, 8, 10, 12, 15, 20 and returns best silhouette.
ClusterResult analyze_clusters(const ActivationData& data);

// Analysis 5: Weight distribution statistics
WeightStats analyze_weight_distribution(SNNNetwork& network);

// Print full analysis report to stdout
void print_analysis_report(
    const std::string& label,
    const ActivationData& act,
    const PowerLawResult& pl,
    const ChiSquareResult& chi,
    const ClusterResult& clu,
    const WeightStats& ws
);

// Save PCA coordinates + cluster assignments to CSV for plotting
void save_cluster_csv(
    const std::string& path,
    const ClusterResult& clu
);

#endif // SNN_STAGE2_ANALYZER_CUH
