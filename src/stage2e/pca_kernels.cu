// =============================================================================
// Stage 2e PCA 增量学习与反投影 kernel 实现 (Task 1)
// =============================================================================
// 设计要点:
//   - pca_update_kernel: Oja's rule 在线学习, 每 PCA_UPDATE_INTERVAL 步调用
//       每个 block 处理一个主成分 k (共 K 个 block), 两阶段:
//         阶段1 strided 累加 + shared memory tree reduction 求投影 proj[k]
//         阶段2 用 proj[k] 更新 W[:,k]
//   - pca_encode_kernel: 提取 PCA 签名 (单 block, 每线程一个 k 做点积),
//       shared memory tree reduction 求 ||sig|| 做 L2 归一化
//   - pca_back_project_kernel: 全量反投影, 每线程一个神经元 i,
//       signature[K] 预加载 shared memory 供全 block 共享, 减少 global 读
//
// 矩阵布局: row-major [N × K], W[i][k] = d_pca_W[i * K + k]
//   阶段1 投影为列访问 (固定 k, 变化 i), stride=K, warp 内 32 线程读同一行的
//   32 个连续 k 时合并访问; 反投影为行访问 (固定 i, 变化 k), 天然合并访问
// =============================================================================

#include "pca_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

namespace stage2e {

// encode kernel 的 block 大小: 需 >= n_components 且为 2 的幂 (tree reduction 要求)
// PCA_N_COMPONENTS = 50, 故取 64
#define PCA_ENCODE_BLOCK_THREADS 64

// =============================================================================
// Task 1.1: PCA 增量学习 (Oja's rule 在线更新)
// =============================================================================
//   W[:,k] += η · (x - W[:,k]·proj) · proj,  x = fr - mean,  proj = W[:,k]ᵀ·x
//
// 启动配置: <<<K, THREADS_PER_BLOCK_2E>>>, K = n_components
//   - blockIdx.x = k (主成分索引)
//   - blockDim.x = THREADS_PER_BLOCK_2E = 256 (需为 2 的幂以支持 tree reduction)
//
// 算法:
//   阶段1 (求投影):
//     每个 thread tid 以 stride=blockDim.x 遍历 i ∈ [tid, tid+bs, ...],
//     累加 partial = Σ W[i*K+k] · (fr[i]-mean[i])
//     shared memory tree reduction 得到 proj[k] (全 block 广播)
//   阶段2 (更新权重):
//     每个 thread tid 以 stride=blockDim.x 遍历 i,
//     W[i*K+k] += η · (x[i] - W[i*K+k]·proj) · proj
//
// 复杂度: 每主成分 N 次乘加 (阶段1) + N 次乘加 (阶段2) = 2N 次/主成分
//         总计 2·N·K = 2·50000·50 = 5M 次乘加
// =============================================================================
__global__ void pca_update_kernel(
    float* __restrict__ d_pca_W,
    const float* __restrict__ d_fr_snapshot,
    const float* __restrict__ d_mean_fr,
    float learning_rate,
    int n_neurons,
    int n_components)
{
    const int k = blockIdx.x;             // 当前主成分索引
    if (k >= n_components) return;

    const int tid = threadIdx.x;
    const int bs  = blockDim.x;           // = THREADS_PER_BLOCK_2E (256)

    // shared memory: 阶段1 存部分和用于 tree reduction, 阶段2 存广播的 proj
    __shared__ float s_partial[THREADS_PER_BLOCK_2E];

    // ---------- 阶段1: 计算 proj[k] = Σ_i W[i][k] · (fr[i] - mean[i]) ----------
    float partial = 0.0f;
    for (int i = tid; i < n_neurons; i += bs) {
        float x = d_fr_snapshot[i] - d_mean_fr[i];   // 中心化向量
        partial += d_pca_W[(size_t)i * n_components + k] * x;
    }
    s_partial[tid] = partial;
    __syncthreads();

    // tree reduction (blockDim.x 需为 2 的幂)
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    // s_partial[0] 即为 proj[k], 所有线程通过 shared memory 读取 (无需单独变量)

    const float proj = s_partial[0];
    __syncthreads();   // 确保 proj 读取完成后才允许复用 s_partial

    // ---------- 阶段2: W[i][k] += η · (x[i] - W[i][k]·proj) · proj ----------
    for (int i = tid; i < n_neurons; i += bs) {
        size_t idx = (size_t)i * n_components + k;
        float w = d_pca_W[idx];
        float x = d_fr_snapshot[i] - d_mean_fr[i];
        // Oja's rule: 投影方向增强, 同时减去正比于当前权重的项 (防发散)
        d_pca_W[idx] = w + learning_rate * (x - w * proj) * proj;
    }
}

// =============================================================================
// Task 1.2: PCA 签名提取
// =============================================================================
//   signature[k] = Σ_i W[i][k] · (fr[i] - mean_fr[i]),  然后 L2 归一化
//
// 启动配置: <<<1, PCA_ENCODE_BLOCK_THREADS>>>, 64 threads (>= 50, 2 的幂)
//   - 每线程 tid 对应一个主成分 k = tid (k < n_components)
//   - 每线程独立遍历 N 个神经元做点积
//   - shared memory + tree reduction 求 Σ sig² → ||sig||, 再逐线程归一化
//
// 复杂度: 每线程 N 次乘加, 总计 N·K = 50000·50 = 2.5M 次乘加
//         单 block 低占用率, 但 K=50 维签名计算量小, ~0.01ms 量级可接受
// =============================================================================
__global__ void pca_encode_kernel(
    const float* __restrict__ d_pca_W,
    const float* __restrict__ d_fr_snapshot,
    const float* __restrict__ d_mean_fr,
    float* __restrict__ d_signature,
    int n_neurons,
    int n_components)
{
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;     // = PCA_ENCODE_BLOCK_THREADS (64)

    // s_sig[tid]    : 原始签名值 (供归一化时读取)
    // s_reduce[tid] : 平方和归约 scratch (tree reduction 用)
    __shared__ float s_sig[PCA_ENCODE_BLOCK_THREADS];
    __shared__ float s_reduce[PCA_ENCODE_BLOCK_THREADS];
    __shared__ float s_norm;              // ||sig||

    // ---------- 阶段1: 每线程计算一个主成分的点积 ----------
    float sig = 0.0f;
    if (tid < n_components) {
        for (int i = 0; i < n_neurons; ++i) {
            float x = d_fr_snapshot[i] - d_mean_fr[i];
            sig += d_pca_W[(size_t)i * n_components + tid] * x;
        }
        s_sig[tid]    = sig;
        s_reduce[tid] = sig * sig;        // 平方, 用于求 ||sig||
    } else {
        s_sig[tid]    = 0.0f;
        s_reduce[tid] = 0.0f;             // 补零以保持 tree reduction 正确性
    }
    __syncthreads();

    // ---------- 阶段2: tree reduction 求 Σ sig² ----------
    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_norm = sqrtf(s_reduce[0]);
    }
    __syncthreads();

    // ---------- 阶段3: L2 归一化写回 ----------
    // 防 0 除: 若 ||sig|| 过小 (如网络静默期), 输出 0 向量
    const float norm = s_norm;
    if (tid < n_components) {
        d_signature[tid] = (norm > 1e-10f) ? (s_sig[tid] / norm) : 0.0f;
    }
}

// =============================================================================
// Task 1.3: PCA 全量反投影
// =============================================================================
//   reconstructed[i] = mean_fr[i] + Σ_k signature[k] · W[i][k]
//
// 启动配置: <<<ceil(N / 256), 256>>>
//   - 每线程一个神经元 i
//   - signature[K] 预加载到 shared memory (K=50 很小, 全 block 共享, 避免每线程
//     重复从 global memory 读取 50 次)
//   - 行访问 W[i*K + k] (固定 i, 变化 k) 天然合并访问
//
// 复杂度: 每线程 K 次乘加, 总计 N·K = 2.5M 次乘加
// =============================================================================
__global__ void pca_back_project_kernel(
    const float* __restrict__ d_pca_W,
    const float* __restrict__ d_mean_fr,
    const float* __restrict__ d_signature,
    float* __restrict__ d_reconstructed,
    int n_neurons,
    int n_components)
{
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // 把 signature[K] 加载到 shared memory (协同加载, 跨线程广播读取无冲突)
    __shared__ float s_sig[PCA_ENCODE_BLOCK_THREADS];  // 64 >= K=50
    for (int kk = tid; kk < n_components; kk += bs) {
        s_sig[kk] = d_signature[kk];
    }
    __syncthreads();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    // 反投影: mean + Σ_k sig[k] · W[i][k]
    float recon = d_mean_fr[i];
    const size_t base = (size_t)i * n_components;
    #pragma unroll 8
    for (int k = 0; k < n_components; ++k) {
        recon += s_sig[k] * d_pca_W[base + k];
    }
    d_reconstructed[i] = recon;
}

// =============================================================================
// Host 端 wrapper 函数
// =============================================================================
// 接受裸指针 (不依赖 MemoryAllocator), 由 scheduler 在集成时调用:
//   - n = N_ASSOCIATION_NEURONS_2E (联合皮层神经元数)
//   - k = PCA_N_COMPONENTS (主成分数 = 50)
//   - lr = PCA_LEARNING_RATE (η_pca = 0.01)
// =============================================================================

void launch_pca_update(float* d_pca_W, const float* d_fr, const float* d_mean,
                       float lr, int n, int k)
{
    // 每个主成分一个 block (grid = K 个 block)
    pca_update_kernel<<<k, THREADS_PER_BLOCK_2E>>>(
        d_pca_W, d_fr, d_mean, lr, n, k);
    CUDA_CHECK_LAST_2E();
}

void launch_pca_encode(const float* d_pca_W, const float* d_fr, const float* d_mean,
                       float* d_sig, int n, int k)
{
    // 单 block, 64 threads (>= n_components=50, 2 的幂以支持 tree reduction)
    pca_encode_kernel<<<1, PCA_ENCODE_BLOCK_THREADS>>>(
        d_pca_W, d_fr, d_mean, d_sig, n, k);
    CUDA_CHECK_LAST_2E();
}

void launch_pca_back_project(const float* d_pca_W, const float* d_mean,
                             const float* d_sig, float* d_recon, int n, int k)
{
    // 每线程一个神经元, 行访问合并
    int blocks = (n + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    pca_back_project_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        d_pca_W, d_mean, d_sig, d_recon, n, k);
    CUDA_CHECK_LAST_2E();
}

} // namespace stage2e
