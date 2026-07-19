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
// 实现：
//   - total_spikes / excitatory_spikes / inhibitory_spikes：atomicAdd int
//   - mean_fire_rate：累加 neurons[idx].fire_rate 到 stats->mean_fire_rate
//     （当 sum 用），finalize 阶段除以 n_neurons
//   - mean_weight：当 d_weights != nullptr 时累加 weights[idx] 到
//     stats->mean_weight（当 sum 用），finalize 阶段除以 n_synapses
//
// 注意：atomicAdd 在 grid 内跨 block 不保证顺序，所以"sum → 除法"必须
// 分两个 kernel：compute_stats_kernel 累加 sum，finalize_stats_kernel
// 做除法。同一 stream 内 kernel 顺序执行，所以 finalize 在 compute
// 之后启动即可保证所有 atomicAdd 已完成。
// -----------------------------------------------------------------------------
__global__ void compute_stats_kernel(
    const NeuronState* neurons,
    const bool* spikes,
    const float* weights,        // 可为 nullptr（向后兼容）
    NetworkStats* stats,
    int n_neurons,
    int n_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 of block 0 清零累加器（只在一个 block 里清即可，
    // 因为后续 __threadfence + 所有 block 都看到清零后才开始累加——
    // 但更安全的做法是 host 端预先 cudaMemset 清零。
    // 这里保留单 block 清零 + threadfence 的写法以兼容 stage0 行为。）
    if (idx == 0) {
        stats->total_spikes = 0;
        stats->excitatory_spikes = 0;
        stats->inhibitory_spikes = 0;
        stats->mean_fire_rate = 0.0f;   // 临时当 sum 用
        stats->mean_weight = 0.0f;      // 临时当 sum 用
    }
    __syncthreads();
    __threadfence();   // 让其他 block 看到 block 0 的清零

    // 累加 spike 计数 + fire_rate sum
    if (idx < n_neurons) {
        if (spikes[idx]) {
            atomicAdd(&stats->total_spikes, 1);
            if (neurons[idx].type == NeuronType::EXCITATORY) {
                atomicAdd(&stats->excitatory_spikes, 1);
            } else {
                atomicAdd(&stats->inhibitory_spikes, 1);
            }
        }
        atomicAdd(&stats->mean_fire_rate, neurons[idx].fire_rate);
    }

    // 累加 weight sum（仅当提供 weights 时）
    if (weights != nullptr && idx < n_synapses) {
        atomicAdd(&stats->mean_weight, weights[idx]);
    }
}

// 单线程 kernel：在 compute_stats_kernel 完成后做除法，把 sum 转成 mean
__global__ void finalize_stats_kernel(NetworkStats* stats,
                                      int n_neurons,
                                      int n_synapses) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n_neurons > 0) {
            stats->mean_fire_rate = stats->mean_fire_rate / (float)n_neurons;
        }
        if (n_synapses > 0) {
            stats->mean_weight = stats->mean_weight / (float)n_synapses;
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
                   NetworkStats* d_stats, int n_neurons,
                   const float* d_weights, int n_synapses) {
    // 选 grid 大小：要覆盖 n_neurons（spike/fire_rate 累加）和 n_synapses
    // （weight 累加）中较大者
    int n_max = (d_weights != nullptr && n_synapses > n_neurons)
              ? n_synapses : n_neurons;
    int blocks = (n_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_stats_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_neurons, d_spikes, d_weights, d_stats, n_neurons, n_synapses);
    CUDA_CHECK_LAST();

    // finalize：单 thread 做除法（同 stream 内顺序执行，所有 atomicAdd 已完成）
    finalize_stats_kernel<<<1, 1>>>(d_stats, n_neurons, n_synapses);
    CUDA_CHECK_LAST();
}
