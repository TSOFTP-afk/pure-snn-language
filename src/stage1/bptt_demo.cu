// =============================================================================
// bptt_demo.cu - BPTT gradient check + minimal learning demo
// =============================================================================
//
// Task: train 10 LIF neurons over 50 time steps to learn a target spike pattern
//
// Loss = (1/2) * sum_t sum_i (S[t,i] - target[t,i])^2
//
// Gradient check:
//   For each weight W[i,j], numeric gradient = (L(W+eps) - L(W-eps)) / (2*eps)
//   Compare with analytic gradient,
//   rel_err = |g_num - g_ana| / (|g_num| + |g_ana| + 1e-8)
//   Pass criterion: rel_err < 1e-3
//
// Note: spike firing is non-differentiable; numeric gradient uses a smooth
//   surrogate to compute loss. This demo uses sigmoid(alpha*(V-theta)) as the
//   differentiable approximation of S.
// =============================================================================

#include "bptt_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// =============================================================================
// BPTT network: stores forward state for all time steps
// =============================================================================
struct BPTTNetwork {
    int N;            // neuron count
    int T;            // time steps
    float beta;
    float threshold;
    float surrogate_alpha;

    // device memory
    float* d_W;                  // [N, N]
    float* d_V_history;          // [T+1, N]  V[0..T]
    float* d_S_history;          // [T+1, N]  S[0..T]
    float* d_I_history;          // [T, N]    I[1..T] (step 0 has no synaptic input)
    float* d_I_ext;              // [T, N]    external input (same or different per step)

    // backward gradients
    float* d_dL_dW;              // [N, N]
    float* d_dL_dV;              // [N]  current dL/dV (full gradient at current step)
};

// =============================================================================
// Utility functions
// =============================================================================
static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void launch_zero(float* d_arr, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    zero_array_kernel<<<blocks, threads>>>(d_arr, n);
    cudaDeviceSynchronize();
}

// =============================================================================
// Forward: run T steps, save all V/S history
// use_smooth controls whether S uses differentiable surrogate (for grad check)
// =============================================================================
float run_forward(BPTTNetwork& net, const float* h_target, bool use_smooth) {
    int N = net.N, T = net.T;
    int threads = 256, blocks = (N + threads - 1) / threads;

    // zero V[0], S[0]
    launch_zero(net.d_V_history, N);
    launch_zero(net.d_S_history, N);

    // forward step by step
    for (int t = 0; t < T; t++) {
        float* V_prev = net.d_V_history + t * N;
        float* S_prev = net.d_S_history + t * N;
        float* V_curr = net.d_V_history + (t + 1) * N;
        float* S_curr = net.d_S_history + (t + 1) * N;
        float* I_curr = net.d_I_history + t * N;

        // I[t] = W * S[t-1] + I_ext[t]
        synapse_forward_kernel<<<blocks, threads>>>(
            I_curr, net.d_W, S_prev, N
        );
        cudaDeviceSynchronize();

        // simplified: assume I_ext = 0 (gradient check task needs no external input)

        smooth_forward_step_kernel<<<blocks, threads>>>(
            V_curr, S_curr, V_prev, S_prev, I_curr,
            N, net.beta, net.threshold, net.surrogate_alpha, use_smooth
        );
        cudaDeviceSynchronize();
    }

    // loss = (1/2) * sum (S[T-1] - target)^2  (only final step has loss, simplified)
    std::vector<float> h_S_final(N);
    cuda_check(cudaMemcpy(h_S_final.data(), net.d_S_history + T * N,
                          N * sizeof(float), cudaMemcpyDeviceToHost), "copy S_final");

    // Use double for loss accumulation: gradient check needs high precision because
    // (loss_plus - loss_minus) is small (~1e-5) while loss itself is O(1).
    double loss = 0.0;
    if (h_target != nullptr) {
        for (int i = 0; i < N; i++) {
            double diff = static_cast<double>(h_S_final[i]) - static_cast<double>(h_target[i]);
            loss += 0.5 * diff * diff;
        }
    }
    return static_cast<float>(loss);
}

// =============================================================================
// Backward: propagate gradients from t = T-1 down to 0
//
// Math summary (see bptt_kernels.cu header for full derivation):
//   Init: dL/dV[T] = (S[T] - target) * sigma'(alpha * (V[T] - theta))
//   For t = T-1 ... 0:
//     dL/dW[i, j] += dL/dV[t+1, i] * S[t, j]
//     v_grad[i] = dL/dV[t+1, i] * beta * (1 - S[t, i])
//     dL_dS_via_W[j] = sum_i dL/dV[t+1, i] * W[i, j]
//     dL/dV[t, i] = v_grad[i] + dL_dS_via_W[i] * sigma'(alpha * (V[t, i] - theta))
// =============================================================================
void run_backward(BPTTNetwork& net, const float* h_target) {
    int N = net.N, T = net.T;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // zero gradients
    launch_zero(net.d_dL_dW, N * N);
    launch_zero(net.d_dL_dV, N);

    // init: dL/dV[T] = (S[T] - target) * sigma'(alpha * (V[T] - theta))
    // (no future gradient at final step)
    {
        std::vector<float> h_S_final(N), h_V_final(N), h_dL_dV(N);
        cuda_check(cudaMemcpy(h_S_final.data(), net.d_S_history + T * N,
                              N * sizeof(float), cudaMemcpyDeviceToHost),
                   "copy S_final for init");
        cuda_check(cudaMemcpy(h_V_final.data(), net.d_V_history + T * N,
                              N * sizeof(float), cudaMemcpyDeviceToHost),
                   "copy V_final for init");
        for (int i = 0; i < N; i++) {
            float dL_dS = h_S_final[i] - h_target[i];
            float x = net.surrogate_alpha * (h_V_final[i] - net.threshold);
            float sigma = 1.0f / (1.0f + expf(-x));
            // dS/dV = alpha * sigma(x) * (1 - sigma(x))  (chain rule on S = sigma(alpha*(V-theta)))
            float dS_dV = net.surrogate_alpha * sigma * (1.0f - sigma);
            h_dL_dV[i] = dL_dS * dS_dV;
        }
        cuda_check(cudaMemcpy(net.d_dL_dV, h_dL_dV.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice), "init dL_dV[T]");
    }

    // temp buffers
    float *d_v_grad, *d_dL_dS_via_W, *d_s_grad_via_V_reset;
    cuda_check(cudaMalloc(&d_v_grad, N * sizeof(float)), "malloc v_grad");
    cuda_check(cudaMalloc(&d_dL_dS_via_W, N * sizeof(float)), "malloc dL_dS_via_W");
    cuda_check(cudaMalloc(&d_s_grad_via_V_reset, N * sizeof(float)), "malloc s_grad_via_V_reset");

    // backward from t = T-1 down to 0
    // dL_dV currently holds dL/dV[t+1]_total at iteration t
    for (int t = T - 1; t >= 0; t--) {
        float* V_prev = net.d_V_history + t * N;   // V[t]
        float* S_prev = net.d_S_history + t * N;   // S[t]

        // 1. accumulate dL/dW and compute V-channel + V-reset-S-channel gradients
        backward_step_main_kernel<<<blocks, threads>>>(
            d_v_grad, d_s_grad_via_V_reset, net.d_dL_dW,
            net.d_dL_dV, V_prev, S_prev, N, net.beta
        );
        cudaDeviceSynchronize();

        if (t > 0) {
            // 2. S-channel via W: dL_dS_via_W[j] = sum_i dL/dV[t+1, i] * W[i, j]
            compute_grad_S_prev_kernel<<<blocks, threads>>>(
                d_dL_dS_via_W, net.d_dL_dV, net.d_W, N
            );
            cudaDeviceSynchronize();

            // 3. combine: dL/dV[t] = v_grad + (dL_dS_via_W + s_grad_via_V_reset) * dS/dV
            combine_final_grad_kernel<<<blocks, threads>>>(
                net.d_dL_dV, d_v_grad, d_dL_dS_via_W, d_s_grad_via_V_reset, V_prev,
                N, net.threshold, net.surrogate_alpha
            );
            cudaDeviceSynchronize();
        }
        // t == 0: no previous V, just stop (dL/dW already accumulated above).
    }

    cudaFree(d_v_grad);
    cudaFree(d_dL_dS_via_W);
    cudaFree(d_s_grad_via_V_reset);
}

// =============================================================================
// Host-side double-precision forward (for gradient check only).
// Replicates the smooth-mode LIF dynamics in double precision on the CPU so
// that the numeric gradient (loss_plus - loss_minus) is not corrupted by float
// rounding (loss differences are ~1e-6 while loss itself is O(1)).
// =============================================================================
static double run_forward_double(BPTTNetwork& net, const float* h_target) {
    int N = net.N, T = net.T;

    std::vector<float> h_W_float(N * N);
    cuda_check(cudaMemcpy(h_W_float.data(), net.d_W, N * N * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy W (double fwd)");
    std::vector<double> h_W(N * N);
    for (int i = 0; i < N * N; i++) h_W[i] = h_W_float[i];

    std::vector<double> V_prev(N, 0.0), S_prev(N, 0.0);
    std::vector<double> V_curr(N), S_curr(N), I(N);

    for (int t = 0; t < T; t++) {
        // I[i] = sum_j W[i, j] * S_prev[j]
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) sum += h_W[i * N + j] * S_prev[j];
            I[i] = sum;
        }
        // V[i] = beta * V_prev[i] * (1 - S_prev[i]) + I[i]
        // S[i] = sigmoid(alpha * (V[i] - theta))
        for (int i = 0; i < N; i++) {
            V_curr[i] = net.beta * V_prev[i] * (1.0 - S_prev[i]) + I[i];
            double x = net.surrogate_alpha * (V_curr[i] - net.threshold);
            S_curr[i] = 1.0 / (1.0 + std::exp(-x));
        }
        std::swap(V_prev, V_curr);
        std::swap(S_prev, S_curr);
    }

    double loss = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = S_prev[i] - static_cast<double>(h_target[i]);
        loss += 0.5 * diff * diff;
    }
    return loss;
}

// =============================================================================
// Gradient check: numeric vs analytic gradient for each weight
// =============================================================================
bool gradient_check(BPTTNetwork& net, const float* h_target, float epsilon = 1e-3f) {
    int N = net.N;
    std::vector<float> h_W(N * N);
    cuda_check(cudaMemcpy(h_W.data(), net.d_W, N * N * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy W for gradcheck");

    // 1. analytic gradient
    run_forward(net, h_target, true /* use_smooth */);
    run_backward(net, h_target);
    std::vector<float> h_analytic_grad(N * N);
    cuda_check(cudaMemcpy(h_analytic_grad.data(), net.d_dL_dW,
                          N * N * sizeof(float), cudaMemcpyDeviceToHost),
               "copy analytic grad");

    // 2. numeric gradient using host-side double-precision forward
    std::vector<double> h_numeric_grad(N * N);
    int total_checked = 0;
    int max_check = 20;

    for (int idx = 0; idx < N * N && total_checked < max_check; idx++) {
        double orig = static_cast<double>(h_W[idx]);

        h_W[idx] = static_cast<float>(orig + epsilon);
        cuda_check(cudaMemcpy(net.d_W, h_W.data(), N * N * sizeof(float),
                              cudaMemcpyHostToDevice), "copy W+");
        double loss_plus = run_forward_double(net, h_target);

        h_W[idx] = static_cast<float>(orig - epsilon);
        cuda_check(cudaMemcpy(net.d_W, h_W.data(), N * N * sizeof(float),
                              cudaMemcpyHostToDevice), "copy W-");
        double loss_minus = run_forward_double(net, h_target);

        h_W[idx] = static_cast<float>(orig);  // restore

        h_numeric_grad[idx] = (loss_plus - loss_minus) / (2.0 * epsilon);
        total_checked++;
    }

    // restore W
    cuda_check(cudaMemcpy(net.d_W, h_W.data(), N * N * sizeof(float),
                          cudaMemcpyHostToDevice), "restore W");

    // 3. compare
    printf("\n=== Gradient check (first %d weights, epsilon=%.0e) ===\n", max_check, epsilon);
    printf("%-8s %-14s %-14s %-12s\n", "idx", "analytic", "numeric", "rel_err");
    int n_pass = 0;
    double max_rel_err = 0.0;
    for (int k = 0; k < max_check; k++) {
        double ga = h_analytic_grad[k];
        double gn = h_numeric_grad[k];
        double rel_err = fabs(ga - gn) / (fabs(ga) + fabs(gn) + 1e-12);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        printf("%-8d %-14.6f %-14.6f %-12.6e\n", k, ga, gn, rel_err);
        if (rel_err < 1e-3) n_pass++;
    }
    printf("\nPass rate: %d/%d, max rel_err: %e\n", n_pass, max_check, max_rel_err);
    printf("Result: %s\n", (n_pass == max_check) ? "PASS" : "FAIL");

    return n_pass == max_check;
}

// =============================================================================
// Minimal learning demo: SGD-train the network to track a target spike pattern
// =============================================================================
void train_demo(BPTTNetwork& net) {
    int N = net.N, T = net.T;
    std::vector<float> h_target(N, 0.0f);

    // target: final step, first 5 neurons fire (S=1), last 5 silent (S=0)
    for (int i = 0; i < 5; i++) h_target[i] = 1.0f;
    for (int i = 5; i < N; i++) h_target[i] = 0.0f;

    printf("\n=== Training demo (target: first 5 fire, last 5 silent) ===\n");
    printf("Initial loss: %f\n", run_forward(net, h_target.data(), false));

    float lr = 0.1f;
    int n_epochs = 200;
    int threads = 256, blocks = (N + threads - 1) / threads;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // forward (smooth, for backward)
        run_forward(net, h_target.data(), true);
        // backward
        run_backward(net, h_target.data());
        // SGD update
        sgd_update_kernel<<<(N * N + 255) / 256, 256>>>(
            net.d_W, net.d_dL_dW, N * N, lr
        );
        cudaDeviceSynchronize();

        if (epoch % 50 == 0 || epoch == n_epochs - 1) {
            // evaluate loss with real spikes
            float real_loss = run_forward(net, h_target.data(), false);
            printf("Epoch %3d: smooth_loss=%f, real_loss=%f\n",
                   epoch, run_forward(net, h_target.data(), true), real_loss);
        }
    }

    // final output
    std::vector<float> h_S_final(N);
    cuda_check(cudaMemcpy(h_S_final.data(), net.d_S_history + T * N,
                          N * sizeof(float), cudaMemcpyDeviceToHost), "final S");
    printf("\nFinal spike state: ");
    for (int i = 0; i < N; i++) printf("%.2f ", h_S_final[i]);
    printf("\nTarget:            ");
    for (int i = 0; i < N; i++) printf("%.2f ", h_target[i]);
    printf("\n");
}

// =============================================================================
// Main entry (called by main.cpp)
// =============================================================================
void run_bptt_demo() {
    printf("=== BPTT gradient check + learning demo ===\n");
    printf("Config: N=10 neurons, T=50 time steps\n");

    BPTTNetwork net;
    net.N = 10;
    net.T = 50;
    net.beta = 0.95f;
    net.threshold = 1.0f;
    net.surrogate_alpha = 5.0f;  // steepness, larger -> closer to real step

    int N = net.N, T = net.T;

    // allocate device memory
    cuda_check(cudaMalloc(&net.d_W, N * N * sizeof(float)), "malloc W");
    cuda_check(cudaMalloc(&net.d_V_history, (T + 1) * N * sizeof(float)), "malloc V");
    cuda_check(cudaMalloc(&net.d_S_history, (T + 1) * N * sizeof(float)), "malloc S");
    cuda_check(cudaMalloc(&net.d_I_history, T * N * sizeof(float)), "malloc I");
    cuda_check(cudaMalloc(&net.d_dL_dW, N * N * sizeof(float)), "malloc dL_dW");
    cuda_check(cudaMalloc(&net.d_dL_dV, N * sizeof(float)), "malloc dL_dV");

    // init W: small random values
    std::vector<float> h_W(N * N);
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_W[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;  // [-0.2, 0.2]
    }
    cuda_check(cudaMemcpy(net.d_W, h_W.data(), N * N * sizeof(float),
                          cudaMemcpyHostToDevice), "init W");

    // gradient check
    std::vector<float> h_target(N);
    for (int i = 0; i < N; i++) h_target[i] = (i < 5) ? 1.0f : 0.0f;
    bool grad_ok = gradient_check(net, h_target.data());

    if (grad_ok) {
        // reinit W (new seed, avoid grad check contamination)
        srand(123);
        for (int i = 0; i < N * N; i++) {
            h_W[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;
        }
        cuda_check(cudaMemcpy(net.d_W, h_W.data(), N * N * sizeof(float),
                              cudaMemcpyHostToDevice), "reinit W");
        train_demo(net);
    } else {
        printf("\n[WARN] Gradient check failed, skipping training demo. Check BPTT implementation.\n");
    }

    // cleanup
    cudaFree(net.d_W);
    cudaFree(net.d_V_history);
    cudaFree(net.d_S_history);
    cudaFree(net.d_I_history);
    cudaFree(net.d_dL_dW);
    cudaFree(net.d_dL_dV);
}
