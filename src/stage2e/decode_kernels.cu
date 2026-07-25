// =============================================================================
// Stage 2e 在线解码 kernel 与预测误差驱动学习实现 (Task 4-5)
// =============================================================================
// 设计要点:
//   - decode_forward_kernel: 256 threads (每线程一字节), 1 block
//       shared memory 分块缓存 spike_flags, 让 256 个线程广播读取同一段
//       突触权重按行主序 (N × 256), 相邻线程 (b, b+1, ...) 读相邻 float,
//       warp 内 32 个连续读 = 128B = 1 cache line, 高效合并访问
//   - decode_softmax_kernel / argmax / error: 单 block 256 threads,
//       用 shared memory tree reduction (log2(256) = 8 步)
//   - decode_weight_update_kernel: 每线程一神经元, 跳过 spike_flags[i]=false
//       (网络中绝大多数神经元不发放, 跳过可省 256 次读写)
//   - decode_weight_normalize_kernel: 每神经元一 block, 256 threads 做 L2 归一化
// =============================================================================

#include "decode_kernels.cuh"
#include <cstdio>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// 文件作用域 device 暂存缓冲: cross-entropy loss 标量 (1 float)
// =============================================================================
// 仿照 modulatory_kernels.cu 的 d_v_scratch 模式: 懒分配, 单次分配后复用
// (loss 是标量, 静态分配 1 个 float 可接受; PersistentBuffers 未为其分配独立 slot)
// =============================================================================
namespace {
float* d_loss_scratch = nullptr;

inline void ensure_loss_scratch() {
    if (d_loss_scratch == nullptr) {
        CUDA_CHECK_2E(cudaMalloc(&d_loss_scratch, sizeof(float)));
        CUDA_CHECK_2E(cudaMemset(d_loss_scratch, 0, sizeof(float)));
    }
}
} // anonymous namespace

// =============================================================================
// Task 4.1: 前向解码 kernel
// =============================================================================
//   logits[b] = Σ_i W_decode[i*256+b] · spike_flags[i]
//
// 启动配置: <<<1, 256>>>
//   - 1 block, 256 threads (每线程负责一个字节 b ∈ [0, 256))
//   - shared memory: TILE_SIZE 个 bool, 用于缓存当前 tile 的 spike_flags
//
// 算法:
//   for tile in n_neurons / TILE_SIZE:
//     1. 256 个线程协同加载 tile 段的 spike_flags 到 shared memory
//     2. __syncthreads()
//     3. 每线程 b 累加 Σ_i W[(tile_start+i)*256 + b] * s_spikes[i]
//        (s_spikes[i] 是 bool, 为 false 时跳过乘法, 大多数神经元不发放)
//     4. __syncthreads()
//   写回 logits[b] = sum
//
// 性能: 60K 神经元 / 256 tile_size = 235 个 tile, 每 tile 256*256 = 64K 乘加
//       总计 ~15.36M 乘加, 单 block 串行执行约 ~1ms (RTX 3060)
//       (可优化为多 block + atomicAdd, 但当前规模可接受)
// =============================================================================
__global__ void decode_forward_kernel(
    const float* __restrict__ decode_weights,
    const bool* __restrict__ spike_flags,
    float* __restrict__ logits,
    int n_neurons)
{
    const int b = threadIdx.x;          // 字节索引 [0, 256)
    const int TILE = 256;               // shared memory tile 大小
    __shared__ bool s_spikes[TILE];     // 缓存当前 tile 的 spike_flags

    float sum = 0.0f;

    for (int tile_start = 0; tile_start < n_neurons; tile_start += TILE) {
        int tile_end = min(tile_start + TILE, n_neurons);
        int tile_size = tile_end - tile_start;

        // 1. 协同加载 spike_flags tile 到 shared memory
        //    256 线程加载 ≤ 256 个 bool, 1:1 映射
        for (int i = b; i < tile_size; i += blockDim.x) {
            s_spikes[i] = spike_flags[tile_start + i];
        }
        __syncthreads();

        // 2. 每线程 b 累加该 tile 内所有发放神经元的权重
        //    权重矩阵按行主序: W[i*256 + b], 相邻 b 在同一行内连续
        //    warp (32 threads) 读取 W[(tile_start+i)*256 + 0..31] = 128B = 1 cache line
        //    (高效合并访问; s_spikes[i] 在 shared memory 广播无冲突)
        #pragma unroll 4
        for (int i = 0; i < tile_size; ++i) {
            if (s_spikes[i]) {
                sum += decode_weights[(size_t)(tile_start + i) * 256 + b];
            }
        }
        __syncthreads();
    }

    // 3. 写回 logits (此时是未归一化的原始 logits, softmax 在另一 kernel)
    logits[b] = sum;
}

// =============================================================================
// Task 4.2: 数值稳定的 in-place softmax
// =============================================================================
//   p[b] = exp(logits[b] - max) / Σ_b' exp(logits[b'] - max)
//
// 启动配置: <<<1, 256>>>
// 算法:
//   1. 256-thread tree reduction 求 max (数值稳定)
//   2. exp(logits[b] - max) 写回 shared memory
//   3. tree reduction 求 sum
//   4. logits[b] = exp_val / sum (in-place 改写为概率)
// =============================================================================
__global__ void decode_softmax_kernel(float* logits, int n)
{
    const int tid = threadIdx.x;
    if (tid >= n) return;

    __shared__ float s_data[256];
    __shared__ float s_max;
    __shared__ float s_sum;

    // 1. 求 max (数值稳定)
    float my_val = logits[tid];
    s_data[tid] = my_val;
    __syncthreads();

    // tree reduction: 取较大值
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < n) {
            if (s_data[tid + stride] > s_data[tid]) {
                s_data[tid] = s_data[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) s_max = s_data[0];
    __syncthreads();

    // 2. exp(logits[b] - max)
    float my_exp = expf(my_val - s_max);
    s_data[tid] = my_exp;
    __syncthreads();

    // 3. 求 sum
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < n) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) s_sum = s_data[0];
    __syncthreads();

    // 4. 归一化 (in-place: logits 现在存的是 softmax 概率)
    logits[tid] = my_exp / (s_sum + 1e-10f);
}

// =============================================================================
// Task 4.3: argmax reduction
// =============================================================================
//   predicted_byte = argmax_b logits[b]
//
// 启动配置: <<<1, 256>>>
// 算法: tree reduction, 同时跟踪 value 和 index
// =============================================================================
__global__ void decode_argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ predicted_byte)
{
    const int tid = threadIdx.x;
    __shared__ float s_vals[256];
    __shared__ int   s_idxs[256];

    s_vals[tid] = logits[tid];
    s_idxs[tid] = tid;
    __syncthreads();

    // tree reduction: 保留较大的 (value, index) 对
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // tid 0 写回结果
    if (tid == 0) {
        predicted_byte[0] = s_idxs[0];
    }
}

// =============================================================================
// Task 5.1: 计算预测误差 + cross-entropy loss
// =============================================================================
//   error[b] = softmax_prob[b] - one_hot(b == target_byte)
//   loss     = -log(softmax_prob[target_byte] + ε)
//
// 启动配置: <<<1, 256>>>
// 算法:
//   - 每线程 b 计算 error[b] = p[b] - (b == target ? 1 : 0)
//   - 线程 0 计算 loss 并写入 loss_output[0]
// =============================================================================
__global__ void decode_error_kernel(
    const float* __restrict__ logits,    // softmax 后的概率
    float* __restrict__ error,
    uint8_t target_byte,
    float* __restrict__ loss_output)
{
    const int tid = threadIdx.x;

    // 误差 = 预测概率 - one-hot(target)
    float target_one_hot = (tid == (int)target_byte) ? 1.0f : 0.0f;
    error[tid] = logits[tid] - target_one_hot;

    // 线程 0 计算 cross-entropy loss = -log(p[target])
    if (tid == 0) {
        float p_target = logits[(int)target_byte];
        // ε = 1e-10 防 log(0)
        loss_output[0] = -logf(p_target + 1e-10f);
    }
}

// =============================================================================
// Task 5.2: 解码权重更新
// =============================================================================
//   ΔW_decode[i*256+b] = -η · error[b] · spike_flags[i]
//
// 启动配置: <<<ceil(N/256), 256>>>
//   - 每线程一神经元 i
//   - 跳过 spike_flags[i]=false 的神经元 (典型 ~99% 不发放)
//   - 对发放神经元, 整行 256 个权重都更新 (因为 spike_flags[i]=1, 公式简化)
//
// 优化思路:
//   - 不用 atomicAdd: 每神经元一行, 线程间无冲突
//   - 跳过 false 神经元: 网络稀疏发放 (~1%), 大幅减少无效读写
// =============================================================================
__global__ void decode_weight_update_kernel(
    float* __restrict__ decode_weights,
    const float* __restrict__ error,
    const bool* __restrict__ spike_flags,
    float learning_rate,
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    // 仅更新发放神经元的解码权重 (spike_flags[i]=true)
    // (未发放神经元 spike_flags[i]=0, ΔW=0, 跳过)
    if (!spike_flags[i]) return;

    // 整行更新: W[i*256 + b] -= η · error[b]  (spike_flags[i]=1)
    float* row = decode_weights + (size_t)i * 256;
    #pragma unroll 4
    for (int b = 0; b < 256; ++b) {
        row[b] -= learning_rate * error[b];
    }
}

// =============================================================================
// Task 5.3: 行 L2 归一化
// =============================================================================
//   每 100 步对 W_decode 每行做 L2 归一化:
//     ||w_i||_2 = sqrt(Σ_b w[i*256+b]²)
//     若 ||w_i||_2 > 1.0: w[i*256+b] /= ||w_i||_2
//
// 启动配置: <<<n_neurons, 256>>>
//   - 每神经元一 block, 256 threads
//   - 256 threads 协同计算该行的平方和, 然后条件缩放
//
// 用途: 防止单个神经元的解码权重行无限增长 (类似权重衰减的正则化效果)
// =============================================================================
__global__ void decode_weight_normalize_kernel(
    float* __restrict__ decode_weights,
    int n_neurons)
{
    const int neuron_idx = blockIdx.x;
    if (neuron_idx >= n_neurons) return;

    const int tid = threadIdx.x;
    float* row = decode_weights + (size_t)neuron_idx * 256;

    // 1. 计算平方和 (256-thread tree reduction)
    __shared__ float s_data[256];
    s_data[tid] = row[tid] * row[tid];
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    const float norm_sq = s_data[0];
    const float norm = sqrtf(norm_sq);

    // 2. 若 L2 范数 > 1.0, 归一化到 1.0
    if (norm > 1.0f && norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        row[tid] *= inv_norm;
    }
}

// =============================================================================
// Host 端 wrapper 实现
// =============================================================================

// Task 4.4: 前向解码链 (forward + softmax + argmax)
//   顺序执行三个 kernel:
//     1. decode_forward_kernel: logits = W · spike_flags (未归一化)
//     2. decode_softmax_kernel: logits = softmax(logits) (in-place 转概率)
//     3. decode_argmax_kernel:  predicted_byte = argmax(logits)
//   全部 1 block × 256 threads, 在默认流上顺序执行
void launch_decode_forward(PersistentBuffers& buf)
{
    // 1. 前向: logits = W · spike_flags
    decode_forward_kernel<<<1, 256>>>(
        buf.d_decode_weights,
        buf.d_spike_flags,
        buf.d_decode_logits,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();

    // 2. softmax (in-place)
    decode_softmax_kernel<<<1, 256>>>(
        buf.d_decode_logits, 256);
    CUDA_CHECK_LAST_2E();

    // 3. argmax → predicted_byte
    decode_argmax_kernel<<<1, 256>>>(
        buf.d_decode_logits,
        buf.d_decode_predicted_byte);
    CUDA_CHECK_LAST_2E();
}

// Task 5.1 host: 计算误差 + 拷贝 loss 到 host
//   顺序: 启动 decode_error_kernel → 同步拷贝 loss 标量到 host
//   (同步拷贝是必要的: 下游 perplexity / accuracy 统计需要立即使用 loss)
void launch_decode_error(PersistentBuffers& buf, uint8_t target_byte, float& out_loss)
{
    ensure_loss_scratch();

    decode_error_kernel<<<1, 256>>>(
        buf.d_decode_logits,
        buf.d_decode_error,
        target_byte,
        d_loss_scratch);
    CUDA_CHECK_LAST_2E();

    // 拷贝 loss 到 host (同步, 因为下游需要立即使用)
    CUDA_CHECK_2E(cudaMemcpy(&out_loss, d_loss_scratch, sizeof(float),
                              cudaMemcpyDeviceToHost));
}

// Task 5.2 host: 权重更新
void launch_decode_weight_update(PersistentBuffers& buf)
{
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    decode_weight_update_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_decode_weights,
        buf.d_decode_error,
        buf.d_spike_flags,
        DECODE_LEARNING_RATE,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// Task 5.3 host: 行 L2 归一化 (每 100 步调用)
void launch_decode_weight_normalize(PersistentBuffers& buf)
{
    // 每个神经元一个 block, 256 threads/block 做 256 元素归一化
    decode_weight_normalize_kernel<<<N_TOTAL_NEURONS_2E, 256>>>(
        buf.d_decode_weights,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

} // namespace stage2e
