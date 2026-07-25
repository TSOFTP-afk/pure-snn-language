#ifndef SNN_STAGE2E_HIPPOCAMPAL_KERNELS_CUH
#define SNN_STAGE2E_HIPPOCAMPAL_KERNELS_CUH

// =============================================================================
// Stage 2e 海马索引编码 kernel (Task 3)
// =============================================================================
// 对应设计文档 v3 强化 C: 50K × HippoIndex 索引表 (12.8 MB)
//
// 职责:
//   1. hippo_encode_kernel: 每 HIPP_ENCODE_INTERVAL 步, 取当前 PCA 签名 (50 维,
//      已 L2 归一化) 与 50K 索引做 cosine 匹配
//        - 命中 (sim >= novelty_threshold): 刷新 importance += 1/(1+replay_count)
//        - 新颖 (sim <  novelty_threshold): 写入 LRU 游标指向的槽位, 推进环形游标
//   2. hippo_get_top_k_kernel: 按 importance 选 top-K (K=HIPP_REPLAY_BATCH=200)
//      索引, 供睡眠重放使用 (partial selection sort + selected bitmap)
//   3. hippo_decay_importance_kernel: 重放后 importance *= HIPP_REPLAY_DECAY,
//      replay_count++
//
// 缓冲区 (在 memory_allocator.cuh 的 PersistentBuffers 中分配):
//   - d_hippo_indices       [HIPP_INDEX_SIZE × HippoIndex]  (12.8 MB, 已有)
//   - d_hippo_write_cursor  [1 × int]   LRU 写入游标 (Task 3 新增, 环形)
//   - d_hippo_filled_count  [1 × int]   已填充条目数 (Task 3 新增, 上限 max_indices)
//   - d_hippo_top_k         [HIPP_REPLAY_BATCH × int]  top-K 索引 (Task 3 新增)
//
// 数学约定:
//   - PCA 签名已 L2 归一化 → cosine_similarity(a, b) = a · b
//   - novelty_threshold 语义为 similarity 阈值 (0.7):
//       sim >= 0.7  → 已有模式 (距离 <= 0.3)
//       sim <  0.7  → 新颖模式 (距离  > 0.3)
//
// 启动约定 (host wrapper 接受裸指针, 由 scheduler 在集成时调用):
//   - launch_hippo_encode:       每 HIPP_ENCODE_INTERVAL 步, 单 block 协作
//   - launch_hippo_get_top_k:    重放前取 top-K, 单 block + selected bitmap
//   - launch_hippo_decay:        重放后衰减, grid 跨步
// =============================================================================

#include "config.h"
#include "types.h"
#include <cuda_runtime.h>

namespace stage2e {

// -----------------------------------------------------------------------------
// CUDA kernel 声明
// -----------------------------------------------------------------------------

// Task 3.1: 海马编码 kernel (单 block, 多线程协作)
//   阶段1: 加载 d_signature[50] 到 shared memory, 分块遍历 50K 索引,
//          每线程计算一个索引的 cosine_similarity = dot(sig, entry.sig),
//          tree reduction 找最大相似度 (最佳匹配) 及对应索引
//   阶段2: 新颖性判定 (单线程写, 避免竞争)
//          - 最佳 sim >= novelty_threshold: 已有模式, importance += 1/(1+replay)
//          - 最佳 sim <  novelty_threshold (或索引表空): 新颖模式, 写入 LRU 槽位,
//            推进 d_write_cursor (环形 % max_indices), 递增 d_filled_count (上限)
//
// 启动配置: <<<1, THREADS_PER_BLOCK_2E=256>>>
//   - 50K / 256 ≈ 195 个索引/线程, 单 block 低占用但 50K 量级可接受 (~0.1ms)
__global__ void hippo_encode_kernel(
    HippoIndex* __restrict__ d_indices,        // [max_indices] 海马索引表
    const float* __restrict__ d_signature,     // [PATTERN_DIM] 当前 PCA 签名 (L2 归一化)
    int* __restrict__ d_write_cursor,          // [1] LRU 写入游标
    int* __restrict__ d_filled_count,          // [1] 已填充条目数
    int current_step,                          // 当前训练步 (写入 pattern_start_step)
    int max_indices,                           // 索引表容量 (= HIPP_INDEX_SIZE = 50000)
    float novelty_threshold);                  // 新颖性相似度阈值 (= HIPP_NOVELTY_THRESHOLD = 0.7)

// Task 3.2: top-K 索引选取 kernel (单 block, partial selection sort)
//   从已填充索引中按 importance 降序选前 K 个, 写入 d_top_k_indices
//   算法: K 轮并行 argmax (selected bitmap 标记已选), bitmap 用 shared memory
//         (HIPP_INDEX_SIZE / 8 = 6250B, 远低于 48KB shared mem 上限)
//   若 filled_count < k, 多余位置写 -1 (无效索引, 调用方需检查)
//
// 启动配置: <<<1, THREADS_PER_BLOCK_2E=256>>>
//   - K=200 轮 × (50000/256 ≈ 195 reads/线程) ≈ 39K reads/线程, ~0.5ms
__global__ void hippo_get_top_k_kernel(
    const HippoIndex* __restrict__ d_indices,  // [max_indices] 海马索引表 (只读)
    int* __restrict__ d_top_k_indices,         // [k] 输出 top-K 索引
    int* __restrict__ d_filled_count,          // [1] 已填充条目数
    int k,                                     // top-K 数 (= HIPP_REPLAY_BATCH = 200)
    int max_indices);                          // 索引表容量

// Task 3.3: 重放后衰减 kernel (grid 跨步)
//   对被重放的 K 个索引: importance *= HIPP_REPLAY_DECAY (0.9), replay_count++
//   生物学: 已重放的记忆痕迹强度衰减, 避免反复重放同一模式 (循环 LRU)
//
// 启动配置: <<<ceil(k / 256), 256>>>
__global__ void hippo_decay_importance_kernel(
    HippoIndex* __restrict__ d_indices,        // [max_indices] 海马索引表
    const int* __restrict__ d_replayed_indices,// [k] 被重放的索引 (可能含 -1, 跳过)
    int k);                                    // 重放批大小

// -----------------------------------------------------------------------------
// Host 端 wrapper 函数 (接受裸指针, 由 scheduler 集成调用)
// -----------------------------------------------------------------------------

// 海马编码: 每 HIPP_ENCODE_INTERVAL 步调用
//   max_idx = HIPP_INDEX_SIZE, novelty_thr = HIPP_NOVELTY_THRESHOLD
void launch_hippo_encode(HippoIndex* d_indices, const float* d_sig,
                         int* d_write_cursor, int* d_filled_count,
                         int step, int max_idx, float novelty_thr);

// top-K 选取: 重放前调用, 输出 top-K 索引到 d_top_k
//   k = HIPP_REPLAY_BATCH, max_idx = HIPP_INDEX_SIZE
void launch_hippo_get_top_k(const HippoIndex* d_indices, int* d_top_k,
                            int* d_filled_count, int k, int max_idx);

// 重放后衰减: 对 d_replayed 中的 K 个索引衰减 importance
//   k = HIPP_REPLAY_BATCH
void launch_hippo_decay(HippoIndex* d_indices, const int* d_replayed, int k);

// P1.2 修复: 时间衰减 — 对所有已填充索引执行 importance *= HIPP_TIME_DECAY
//   调用时机: 每 HIPP_ENCODE_INTERVAL 步, 在 launch_hippo_encode 之后执行
//   效果: 老模式自然衰减, 新模式写入时重置为 1.0, 形成长尾分布
void launch_hippo_time_decay(HippoIndex* d_indices, int filled_count, int max_indices);

// -----------------------------------------------------------------------------
// Task 4-5: 睡眠重放电流注入 kernel + 完整重放周期
// -----------------------------------------------------------------------------

// replay_inject_kernel: 元素级重放电流注入 kernel
//   对每个联合皮层神经元 i, d_replay_injection[i] = d_reconstructed[i] * inject_gain
//   简化方案: 重放注入后由主循环 STDP kernel 自然学习, 本 kernel 仅注入电流
//   d_replay_injection: [n_neurons] in/out (覆盖前一次)
//   d_reconstructed:    [n_neurons] PCA 反投影重建的发放率
//   n_neurons:          N_ASSOCIATION_NEURONS_2E = 50000
//   inject_gain:        REPLAY_INJECT_GAIN = 2.0f
__global__ void replay_inject_kernel(
    float* __restrict__ d_replay_injection,
    const float* __restrict__ d_reconstructed,
    int n_neurons,
    float inject_gain);

// launch_replay_cycle: 睡眠重放完整流程 (自由函数, 由 scheduler.launch_replay 调用)
//   1. launch_hippo_get_top_k: 取 importance top-K 索引
//   2. 对每个 top-K 索引: 提取 signature → PCA 反投影 → replay_inject_kernel
//   3. launch_hippo_decay: 重放后衰减 importance, replay_count++
//   参数均为裸指针 (不依赖 MemoryAllocator), 由调用方分配临时缓冲
//   step 未直接使用, 预留扩展
void launch_replay_cycle(
    HippoIndex* d_indices,
    int* d_top_k_indices,
    int* d_filled_count,
    float* d_replay_injection,
    float* d_sig_buffer,
    const float* d_pca_W,
    const float* d_pca_mean,
    float* d_reconstructed_buffer,
    int step, int max_indices, int batch_size);

} // namespace stage2e

#endif // SNN_STAGE2E_HIPPOCAMPAL_KERNELS_CUH
