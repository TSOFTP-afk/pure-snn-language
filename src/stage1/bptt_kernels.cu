// =============================================================================
// bptt_kernels.cu - BPTT forward + backward CUDA kernels
// =============================================================================
//
// LIF forward (time step dt=1):
//   I[t+1] = W * S[t]            (synapse: post-syn current from pre-syn spikes)
//   V[t+1] = beta * V[t] * (1 - S[t]) + I[t+1]      (reset after firing)
//   S[t+1] = Theta(V[t+1] - theta)                  (step / sigmoid surrogate)
//
// Loss = (1/2) * sum_i (S[T, i] - target[i])^2     (only final step)
//
// Backward (t = T-1 -> 0):
//   At step t we are given dL/dV[t+1]_total (full gradient from future + direct).
//   1. Accumulate weight gradient:
//        dL/dW[i, j] += dL/dV[t+1, i] * S[t, j]
//      (because I[t+1, i] = sum_j W[i, j] * S[t, j], so dI[t+1,i]/dW[i,j] = S[t,j]
//       and dL/dW[i,j] = sum_t dL/dI[t+1,i] * S[t,j] = sum_t dL/dV[t+1,i] * S[t,j])
//   2. V-channel (V[t] -> V[t+1] directly):
//        dL/dV[t]_via_V = dL/dV[t+1] * beta * (1 - S[t])
//      (does NOT get multiplied by sigma'(alpha*(V[t]-theta)))
//   3. S-channel (V[t] -> S[t] -> I[t+1] -> V[t+1]):
//        dL/dS[t, j]_via_W = sum_i dL/dV[t+1, i] * W[i, j]
//        dL/dV[t]_via_S    = dL/dS[t]_via_W * sigma'(alpha * (V[t] - theta))
//      (DOES get multiplied by sigma')
//   4. dL/dV[t]_total = dL/dV[t]_via_V + dL/dV[t]_via_S
//
// Initialization (before the loop, at t = T-1):
//   dL/dS[T]_direct = S[T] - target
//   dL/dV[T]_total  = dL/dS[T]_direct * sigma'(alpha * (V[T] - theta))
//   (no future gradient at the final step)
// =============================================================================

#include "bptt_kernels.cuh"
#include <cmath>

// Forward: LIF one step (smooth surrogate or real spike depending on use_smooth)
__global__ void smooth_forward_step_kernel(
    float* V, float* S,
    const float* V_prev, const float* S_prev, const float* I_input,
    int N, float beta, float threshold, float alpha, bool use_smooth
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float v = beta * V_prev[i] * (1.0f - S_prev[i]) + I_input[i];

    float s;
    if (use_smooth) {
        float x = alpha * (v - threshold);
        s = 1.0f / (1.0f + expf(-x));
    } else {
        s = (v >= threshold) ? 1.0f : 0.0f;
    }

    V[i] = v;
    S[i] = s;
}

// Synapse forward: I = W * S_prev
__global__ void synapse_forward_kernel(
    float* I_out, const float* W, const float* S_prev, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += W[i * N + j] * S_prev[j];
    }
    I_out[i] = sum;
}

// Add constant bias: I[i] += bias[i]
__global__ void add_bias_kernel(float* I, const float* bias, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    I[i] += bias[i];
}

// Backward main step.
//   v_grad[i]             = dL_dV[i] * beta * (1 - S_prev[i])
//   s_grad_via_V_reset[i] = dL_dV[i] * (-beta * V_prev[i])
//   dL_dW[i, j] += dL_dV[i] * S_prev[j]   (atomic)
__global__ void backward_step_main_kernel(
    float* v_grad,
    float* s_grad_via_V_reset,
    float* dL_dW,
    const float* dL_dV,
    const float* V_prev,
    const float* S_prev,
    int N, float beta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float grad = dL_dV[i];
    v_grad[i] = grad * beta * (1.0f - S_prev[i]);
    s_grad_via_V_reset[i] = grad * (-beta * V_prev[i]);

    for (int j = 0; j < N; j++) {
        atomicAdd(&dL_dW[i * N + j], grad * S_prev[j]);
    }
}

// dL_dS_via_W[j] = sum_i dL_dV[i] * W[i, j]
__global__ void compute_grad_S_prev_kernel(
    float* dL_dS_via_W, const float* dL_dV, const float* W, int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += dL_dV[i] * W[i * N + j];
    }
    dL_dS_via_W[j] = sum;
}

// Combine: dL_dV_out[i] = v_grad[i] + (dL_dS_via_W[i] + s_grad_via_V_reset[i]) * dS/dV
//   dS/dV = alpha * sigma(x) * (1 - sigma(x)),  x = alpha * (V_prev[i] - theta)
__global__ void combine_final_grad_kernel(
    float* dL_dV_out,
    const float* v_grad,
    const float* dL_dS_via_W,
    const float* s_grad_via_V_reset,
    const float* V_prev,
    int N, float threshold, float surrogate_alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = surrogate_alpha * (V_prev[i] - threshold);
    float sigma = 1.0f / (1.0f + expf(-x));
    // dS/dV = alpha * sigma(x) * (1 - sigma(x)),  because S = sigma(alpha*(V-theta))
    float dS_dV = surrogate_alpha * sigma * (1.0f - sigma);

    float dL_dS_total = dL_dS_via_W[i] + s_grad_via_V_reset[i];
    dL_dV_out[i] = v_grad[i] + dL_dS_total * dS_dV;
}

// SGD update
__global__ void sgd_update_kernel(
    float* W, const float* dL_dW, int N_total, float lr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_total) return;
    W[idx] -= lr * dL_dW[idx];
}

// Zero out
__global__ void zero_array_kernel(float* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    arr[idx] = 0.0f;
}
