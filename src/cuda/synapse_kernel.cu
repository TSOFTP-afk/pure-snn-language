// =============================================================================
// synapse_kernel.cu - 突触传播 CUDA kernel
// =============================================================================
//
// 突触传播：将突触前神经元的脉冲传到突触后神经元的电流累加器
//
// 数据结构选择：CSR (Compressed Sparse Row)
//   - row_ptr[n+1]：行指针，row_ptr[i] 到 row_ptr[i+1]-1 是指向神经元 i 的突触
//   - col_idx[nnz]：列索引（突触前神经元）
//   - weights[nnz]：权重
//
// 为什么用 CSR 而不是 COO：
//   - CSR 允许每个突触后神经元独立累加电流（atomic 或 per-block reduce）
//   - COO 需要 atomic add，性能差
//   - CSR 是稀疏矩阵-向量乘的标准格式
//
// 两种实现：
//   1. 基础版：一个 thread 处理一个突触，atomicAdd 到 post_current
//   2. CSR 版：一个 block 处理一个突触后神经元，warp reduce 后单次写入
// =============================================================================

#include "synapse.cuh"

// -----------------------------------------------------------------------------
// 基础版：一个 thread 一个突触，atomicAdd
// 简单但有竞争，适合突触稀疏激活时
// -----------------------------------------------------------------------------
__global__ void synaptic_transmission_kernel(
    const Synapse* synapses,
    int n_synapses,
    const bool* pre_spikes,
    float* post_current,
    int n_post_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_synapses) return;

    const Synapse& s = synapses[idx];

    // 如果突触前神经元发放，则向突触后注入电流
    if (pre_spikes[s.pre_idx]) {
        float current = s.weight;
        // 兴奋性突触：weight > 0；抑制性突触：weight < 0
        // weight 的符号在初始化时设定
        atomicAdd(&post_current[s.post_idx], current);
    }
}

// -----------------------------------------------------------------------------
// CSR 优化版：一个 block 处理一个突触后神经元
// 每个 block 内的 threads 分摊该神经元的所有入边，先 local 累加，再写回
// 避免大量 atomic 冲突
// -----------------------------------------------------------------------------
__global__ void synaptic_transmission_csr_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* weights,
    int n_post,
    const bool* pre_spikes,
    float* post_current
) {
    int post_idx = blockIdx.x;
    if (post_idx >= n_post) return;

    int start = row_ptr[post_idx];
    int end = row_ptr[post_idx + 1];
    int n_incoming = end - start;

    if (n_incoming == 0) return;

    // 每个 thread 处理若干突触，先累加到 shared memory
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = 0.0f;
    __syncthreads();

    // 分摊入边到 threads
    for (int i = threadIdx.x; i < n_incoming; i += blockDim.x) {
        int syn_idx = start + i;
        int pre_idx = col_idx[syn_idx];
        if (pre_spikes[pre_idx]) {
            sdata[threadIdx.x] += weights[syn_idx];
        }
    }
    __syncthreads();

    // 树形规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // thread 0 写回结果
    if (threadIdx.x == 0) {
        post_current[post_idx] += sdata[0];
    }
}

// -----------------------------------------------------------------------------
// 清零电流累加器
// -----------------------------------------------------------------------------
__global__ void clear_current_kernel(float* current, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    current[idx] = 0.0f;
}

// -----------------------------------------------------------------------------
// 权重同步：d_synapses_.weight → d_weights_
// STDP 更新的是 d_synapses_，但突触传播 CSR 用 d_weights_
// 每步 STDP 后必须同步一次，否则突触传播用的是旧权重
// 注：由于 network_init.cu 中突触按 post_idx 顺序连续生成，
//     synapses[idx] 与 CSR 的 weights[idx] 索引完全一致
// -----------------------------------------------------------------------------
__global__ void sync_weights_kernel(
    float* __restrict__ weights,
    const Synapse* __restrict__ synapses,
    int n_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_synapses) return;
    weights[idx] = synapses[idx].weight;
}

// =============================================================================
// Host 端包装函数
// =============================================================================

void synaptic_transmission(const Synapse* d_synapses, int n_synapses,
                           const bool* d_pre_spikes,
                           float* d_post_current, int n_post) {
    int blocks = (n_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    synaptic_transmission_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_synapses, n_synapses, d_pre_spikes, d_post_current, n_post);
    CUDA_CHECK_LAST();
}

void synaptic_transmission_csr(const int* d_row_ptr, const int* d_col_idx,
                                const float* d_weights, int n_post,
                                const bool* d_pre_spikes,
                                float* d_post_current) {
    // 一个 block 处理一个突触后神经元
    int blocks = n_post;
    int threads = THREADS_PER_BLOCK;
    int shared_mem = threads * sizeof(float);

    synaptic_transmission_csr_kernel<<<blocks, threads, shared_mem>>>(
        d_row_ptr, d_col_idx, d_weights, n_post, d_pre_spikes, d_post_current);
    CUDA_CHECK_LAST();
}

void clear_current(float* d_current, int n) {
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    clear_current_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_current, n);
    CUDA_CHECK_LAST();
}

void sync_weights(float* d_weights, const Synapse* d_synapses, int n_synapses) {
    int blocks = (n_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sync_weights_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_weights, d_synapses, n_synapses);
    CUDA_CHECK_LAST();
}
