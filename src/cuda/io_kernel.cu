// =============================================================================
// io_kernel.cu - 输入输出辅助 kernel
// =============================================================================

#include "neuron.cuh"
#include "synapse.cuh"
#include "types.h"

// -----------------------------------------------------------------------------
// 将外部输入注入到感觉皮层
// -----------------------------------------------------------------------------
__global__ void inject_input_kernel(
    float* input_current,
    const float* external_input,
    int n_sensory,
    float gain
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sensory) return;
    input_current[idx] += external_input[idx] * gain;
}

// -----------------------------------------------------------------------------
// 从运动皮层读取输出
// -----------------------------------------------------------------------------
__global__ void read_output_kernel(
    const bool* spikes,
    float* output_rate,
    int n_motor,
    int motor_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_motor) return;
    output_rate[idx] = spikes[motor_start + idx] ? 1.0f : 0.0f;
}

// -----------------------------------------------------------------------------
// 统计网络状态（用于监控）
// -----------------------------------------------------------------------------
__global__ void compute_stats_kernel(
    const NeuronState* neurons,
    const bool* spikes,
    NetworkStats* stats,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用 atomic 累加统计
    if (idx == 0) {
        stats->total_spikes = 0;
        stats->excitatory_spikes = 0;
        stats->inhibitory_spikes = 0;
    }
    __syncthreads();

    if (idx >= n_neurons) return;

    if (spikes[idx]) {
        atomicAdd(&stats->total_spikes, 1);
        if (neurons[idx].type == NeuronType::EXCITATORY) {
            atomicAdd(&stats->excitatory_spikes, 1);
        } else {
            atomicAdd(&stats->inhibitory_spikes, 1);
        }
    }
}

// =============================================================================
// Host 端包装函数
// =============================================================================

void inject_input(float* d_input, const float* d_external,
                  int n_sensory, float gain) {
    int blocks = (n_sensory + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    inject_input_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_input, d_external, n_sensory, gain);
    CUDA_CHECK_LAST();
}

void read_output(const bool* d_spikes, float* d_output,
                 int n_motor, int motor_start) {
    int blocks = (n_motor + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    read_output_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_spikes, d_output, n_motor, motor_start);
    CUDA_CHECK_LAST();
}

void compute_stats(const NeuronState* d_neurons, const bool* d_spikes,
                   NetworkStats* d_stats, int n_neurons) {
    int blocks = (n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_stats_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_neurons, d_spikes, d_stats, n_neurons);
    CUDA_CHECK_LAST();
}
