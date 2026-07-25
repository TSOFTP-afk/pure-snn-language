#ifndef SNN_STAGE2E_WM_KERNELS_CUH
#define SNN_STAGE2E_WM_KERNELS_CUH

// =============================================================================
// Stage 2e 工作记忆 (WM) 完整闭环 kernel (Task 9)
// =============================================================================
// 对应设计文档 v3 强化 E: WM 50 槽位 + 前额叶 5K 神经元闭环
//
// 闭环流程:
//   1. wm_write_kernel (每 WM_WRITE_INTERVAL 步):
//      - 接收当前 PCA 签名 (50 维, L2 归一化)
//      - 与 50 个 WM 槽位的 pattern 做 cosine 相似度匹配
//      - 若 max_sim < WM_NOVELTY_THRESHOLD: 新颖模式 → LRU 写入最旧槽位
//      - 否则: 已有模式 → 刷新匹配槽位 (activation = 1.0)
//   2. wm_maintain_kernel (每步):
//      - 衰减: activation *= WM_DECAY, age++
//      - 若 activation > WM_INJECT_THRESHOLD: 内联 PCA 反投影重建发放率,
//        注入到绑定前额叶组 (100 神经元)
//      - 注入电流 = reconstructed[j] × activation × WM_INJECT_GAIN
//
// 缓冲区 (在 memory_allocator.cuh 的 PersistentBuffers 中):
//   - d_wm_slots [50 × WMSlot]           WM 槽位 (已有, v3 强化 E)
//   - d_wm_write_cursor [1]              LRU 写入游标 (Task 9 新增)
//   - d_prefrontal_input [5000]          前额叶输入电流缓冲 (Task 9 新增)
//   - d_pca_W [N × K]                    PCA 基矩阵 (已有, v3 强化 A)
//   - d_mean_fr [N]                      滑动平均发放率 (由调用方提供)
//
// 前额叶布局:
//   全局神经元索引: [0, 50000) = 联合皮层, [50000, 55000) = 前额叶 (50 组 × 100)
//   d_prefrontal_input 索引 [0, 5000) 对应前额叶神经元 [50000, 55000)
//   组 g 占用 d_prefrontal_input[g*100 .. (g+1)*100)
//
// 注意:
//   - 不修改 scheduler.cu 中的 p3_wm_update_kernel (避免冲突, 本文件为独立新增)
//   - PCA 反投影在 wm_maintain_kernel 中内联实现 (避免跨 kernel 调用复杂度)
// =============================================================================

#include "config.h"
#include "types.h"
#include <cuda_runtime.h>

namespace stage2e {

// -----------------------------------------------------------------------------
// CUDA kernel 声明
// -----------------------------------------------------------------------------

// Task 9.1: WM 写入 (新颖检测 + LRU 替换)
//   启动配置: <<<1, n_slots>>>, n_slots = 50 (单 block, 每线程一个槽位)
//   算法:
//     1. 每线程计算 d_signature 与 d_wm_slots[tid].pattern 的 cosine 相似度
//     2. shared memory 串行归约找最大相似度 + 索引 (50 元素, 串行足够)
//     3. 若 max_sim < novelty_threshold: 新颖 → 写入 LRU 游标位置, 游标前进
//     4. 否则: 刷新匹配槽位 (activation = 1.0, age = 0)
//   约束: n_slots <= WM_SLOTS (shared memory 数组大小限制)
__global__ void wm_write_kernel(
    WMSlot* __restrict__ d_wm_slots,           // [n_slots]
    const float* __restrict__ d_signature,     // [WM_PATTERN_DIM] PCA 签名 (L2 归一化)
    int* __restrict__ d_wm_write_cursor,       // [1] LRU 游标
    int current_step,                          // 当前步 (预留扩展)
    int n_slots,                               // 槽位数 (= WM_SLOTS = 50)
    float novelty_threshold);                  // 新颖判定阈值 (= WM_NOVELTY_THRESHOLD = 0.7)

// Task 9.2: WM 维持与注入 (衰减 + PCA 反投影注入前额叶)
//   启动配置: <<<1, n_slots>>>, n_slots = 50 (单 block, 每线程一个槽位)
//   算法:
//     1. activation *= decay_factor, age++
//     2. 若 activation > inject_threshold:
//        - 内联 PCA 反投影: recon[j] = mean_fr[base+j] + Σ_k pattern[k] · W[base+j][k]
//        - 注入: d_prefrontal_input[group_offset + j] += recon[j] × activation × gain
//   注: 调用方需在调用前清零 d_prefrontal_input (本 kernel 仅做累加注入)
__global__ void wm_maintain_kernel(
    WMSlot* __restrict__ d_wm_slots,           // [n_slots]
    const float* __restrict__ d_pca_W,         // [n_neurons × n_pca_components] PCA 基矩阵 (row-major)
    const float* __restrict__ d_mean_fr,       // [n_neurons] 滑动平均发放率
    float* __restrict__ d_prefrontal_input,    // [n_prefrontal] 前额叶输入电流
    int n_slots,                               // 50
    int n_prefrontal,                          // 5000
    int group_size,                            // 100
    float inject_threshold,                    // 注入阈值 (= WM_INJECT_THRESHOLD = 0.3)
    float decay_factor,                        // 衰减因子 (= WM_DECAY = 0.995)
    int n_pca_components,                      // PCA 主成分数 (= WM_PATTERN_DIM = 50)
    int n_neurons);                            // 联合皮层+前额叶 (= 55000, 不含运动皮层)

// -----------------------------------------------------------------------------
// Host 端 wrapper 函数 (接受裸指针, 由 scheduler 集成调用)
// -----------------------------------------------------------------------------

// WM 写入: 新颖检测 + LRU 替换 (每 WM_WRITE_INTERVAL 步调用)
//   n = WM_SLOTS, thr = WM_NOVELTY_THRESHOLD
void launch_wm_write(WMSlot* d_slots, const float* d_sig,
                     int* d_cursor, int step, int n, float thr);

// WM 维持: 衰减 + PCA 反投影注入 (每步调用)
//   n_slots = WM_SLOTS, n_pf = N_PREFRONTAL_NEURONS, group_sz = NEURONS_PER_PF_GROUP
//   inject_thr = WM_INJECT_THRESHOLD, decay = WM_DECAY
//   n_comp = PCA_N_COMPONENTS, n_neurons = N_ASSOCIATION_NEURONS_2E + N_PREFRONTAL_NEURONS
//   注: 调用方需在调用前清零 d_pf_input
void launch_wm_maintain(WMSlot* d_slots, const float* d_pca_W,
                        const float* d_mean_fr, float* d_pf_input,
                        int n_slots, int n_pf, int group_sz,
                        float inject_thr, float decay, int n_comp, int n_neurons);

} // namespace stage2e

#endif // SNN_STAGE2E_WM_KERNELS_CUH
