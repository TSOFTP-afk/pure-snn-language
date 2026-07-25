#ifndef SNN_STAGE2E_PCA_KERNELS_CUH
#define SNN_STAGE2E_PCA_KERNELS_CUH

// =============================================================================
// Stage 2e PCA 增量学习与反投影 kernel (Task 1)
// =============================================================================
// 对应设计文档 v3 强化 A: 全量 GPU 矩阵 PCA 反投影
//
// 数学:
//   中心化向量:   x[i]      = fr[i] - mean_fr[i]
//   投影 (签名):  sig[k]    = Σ_i W[i,k] · x[i]
//   Oja 在线更新: W[i,k]   += η · (x[i] - W[i,k]·sig[k]) · sig[k]
//   反投影重建:   recon[i]  = mean_fr[i] + Σ_k sig[k] · W[i,k]
//
// 矩阵布局: row-major [N × K], W[i][k] = d_pca_W[i * K + k]
//   - N = n_neurons   (联合皮层神经元数, 运行时传入 N_ASSOCIATION_NEURONS_2E)
//   - K = n_components (主成分数, 运行时传入 PCA_N_COMPONENTS = 50)
//
// 缓冲区 (在 memory_allocator.cuh 的 PersistentBuffers 中已分配):
//   - d_pca_W [N_TOTAL_NEURONS_2E × PATTERN_DIM float]  (实际分配 60K×50,
//     足够容纳联合皮层 50K×50 的子矩阵, 当前无任何 kernel 读写, 本文件首次使用)
//
// 启动约定 (host wrapper 接受裸指针, 由 scheduler 在集成时调用):
//   - launch_pca_update:        每 PCA_UPDATE_INTERVAL 步, 50 个 block 各处理一个主成分
//   - launch_pca_encode:        提取当前发放率快照的 PCA 签名 (L2 归一化)
//   - launch_pca_back_project:  从签名全量反投影重建发放率
// =============================================================================

#include "config.h"
#include <cuda_runtime.h>

namespace stage2e {

// -----------------------------------------------------------------------------
// CUDA kernel 声明
// -----------------------------------------------------------------------------

// Task 1.1: PCA 增量学习 (Oja's rule 在线更新)
//   W[:,k] += η · (x - W[:,k]·(W[:,k]ᵀx)) · (W[:,k]ᵀx),  x = fr - mean
// 启动配置: grid = (n_components,), block = (THREADS_PER_BLOCK_2E,)
//   - 每个 block 处理一个主成分 k = blockIdx.x
//   - 阶段1: 跨 N 个神经元 strided 累加 + shared memory tree reduction 求投影 proj[k]
//   - 阶段2: 用广播的 proj[k] 更新该列所有 W[i][k]
__global__ void pca_update_kernel(
    float* __restrict__ d_pca_W,            // [N × K] 基矩阵 (device, row-major)
    const float* __restrict__ d_fr_snapshot,// [N] 当前发放率快照
    const float* __restrict__ d_mean_fr,    // [N] 滑动平均发放率
    float learning_rate,                    // η_pca
    int n_neurons,                          // N (联合皮层神经元数)
    int n_components);                      // K (主成分数)

// Task 1.2: PCA 签名提取
//   signature[k] = Σ_i W[i][k] · (fr[i] - mean_fr[i]), 然后 L2 归一化
// 启动配置: 1 block, PCA_ENCODE_BLOCK_THREADS threads (>= n_components, 2 的幂)
//   - 每线程一个主成分 k, 遍历 N 个神经元做点积
//   - shared memory + tree reduction 计算 ||sig|| 用于 L2 归一化
__global__ void pca_encode_kernel(
    const float* __restrict__ d_pca_W,      // [N × K]
    const float* __restrict__ d_fr_snapshot,// [N]
    const float* __restrict__ d_mean_fr,    // [N]
    float* __restrict__ d_signature,        // [K] 输出签名 (L2 归一化)
    int n_neurons,
    int n_components);

// Task 1.3: PCA 全量反投影
//   reconstructed[i] = mean_fr[i] + Σ_k signature[k] · W[i][k]
// 启动配置: grid = ceil(n_neurons / 256), block = 256
//   - 每线程一个神经元 i, 遍历 K 个主成分累加
//   - signature[K] 预加载到 shared memory 供全 block 共享
__global__ void pca_back_project_kernel(
    const float* __restrict__ d_pca_W,      // [N × K]
    const float* __restrict__ d_mean_fr,    // [N]
    const float* __restrict__ d_signature,  // [K]
    float* __restrict__ d_reconstructed,    // [N] 输出重建
    int n_neurons,
    int n_components);

// -----------------------------------------------------------------------------
// Host 端 wrapper 函数 (接受裸指针, 由 scheduler 集成调用)
// -----------------------------------------------------------------------------

// PCA 增量学习: 每 PCA_UPDATE_INTERVAL 步调用 (需 warmup 后才启用)
//   n = N_ASSOCIATION_NEURONS_2E, k = PCA_N_COMPONENTS
void launch_pca_update(float* d_pca_W, const float* d_fr, const float* d_mean,
                       float lr, int n, int k);

// PCA 签名提取: 输出 L2 归一化的 K 维签名
void launch_pca_encode(const float* d_pca_W, const float* d_fr, const float* d_mean,
                       float* d_sig, int n, int k);

// PCA 全量反投影: 从签名重建 N 维发放率
void launch_pca_back_project(const float* d_pca_W, const float* d_mean,
                             const float* d_sig, float* d_recon, int n, int k);

} // namespace stage2e

#endif // SNN_STAGE2E_PCA_KERNELS_CUH
