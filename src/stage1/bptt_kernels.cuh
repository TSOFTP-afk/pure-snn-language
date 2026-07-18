#ifndef SNN_STAGE1_BPTT_KERNELS_CUH
#define SNN_STAGE1_BPTT_KERNELS_CUH

#include <cuda_runtime.h>

// Forward: LIF update for one time step (smooth surrogate or real spike)
__global__ void smooth_forward_step_kernel(
    float* V, float* S,
    const float* V_prev, const float* S_prev, const float* I_input,
    int N, float beta, float threshold, float alpha, bool use_smooth
);

// Synapse forward: I = W * S_prev
__global__ void synapse_forward_kernel(
    float* I_out, const float* W, const float* S_prev, int N
);

// Add constant bias to current: I[i] += bias[i]
__global__ void add_bias_kernel(float* I, const float* bias, int N);

// Backward main step: accumulate dL/dW and compute V-channel gradient for previous step.
//   Inputs:
//     dL_dV[i]   = dL/dV[t+1, i]_total  (full gradient at current step)
//     V_prev[i]  = V[t, i]
//     S_prev[i]  = S[t, i]
//   Outputs:
//     v_grad[i]              = dL/dV[t+1, i] * beta * (1 - S[t, i])   (V-channel part of dL/dV[t])
//     s_grad_via_V_reset[i]  = dL/dV[t+1, i] * (-beta * V[t, i])      (S-channel via V reset term)
//   Side effects:
//     dL_dW[i, j] += dL/dV[t+1, i] * S[t, j]
__global__ void backward_step_main_kernel(
    float* v_grad,
    float* s_grad_via_V_reset,
    float* dL_dW,
    const float* dL_dV,
    const float* V_prev,
    const float* S_prev,
    int N, float beta
);

// Compute S-channel gradient via W: dL_dS_via_W[j] = sum_i dL/dV[t+1, i] * W[i, j]
__global__ void compute_grad_S_prev_kernel(
    float* dL_dS_via_W, const float* dL_dV, const float* W, int N
);

// Combine V-channel and S-channel (both via-W and via-V-reset) into dL/dV[t]_total.
//   dL/dV[t, i] = v_grad[i] + (dL_dS_via_W[i] + s_grad_via_V_reset[i]) * dS/dV
// where dS/dV = alpha * sigma(x) * (1 - sigma(x)),  x = alpha * (V[t, i] - theta).
__global__ void combine_final_grad_kernel(
    float* dL_dV_out,
    const float* v_grad,
    const float* dL_dS_via_W,
    const float* s_grad_via_V_reset,
    const float* V_prev,         // V[t]
    int N, float threshold, float surrogate_alpha
);

// SGD update
__global__ void sgd_update_kernel(
    float* W, const float* dL_dW, int N_total, float lr
);

// Zero out
__global__ void zero_array_kernel(float* arr, int N);

#endif
