#ifndef SNN_STAGE2E_COACTIVATION_KERNELS_CUH
#define SNN_STAGE2E_COACTIVATION_KERNELS_CUH

// =============================================================================
// Stage 2e 共激活跟踪采样 kernel (Task 6)
// =============================================================================
// 对应设计文档 §2.4 (v3 强化 D):
//   - coactivation_sample_kernel: 每步从当前发放神经元中随机采样候选对,
//     更新或创建 CoactTracker 条目 (coact_count / modulator_score / last_seen)
//   - coactivation_prune_kernel: 淘汰长期 (coact_count==0 持续 stale_threshold
//     步) 未活跃的 tracker 条目, 清零槽位
//
// 缓冲区 (由 memory_allocator 分配):
//   - d_coact_trackers [COACT_TRACKER_SIZE × CoactTracker]  (8 MB)
//   - d_tracker_count  [1 × int]   当前已用 tracker 条目数 (append-only)
//   - d_spike_flags    [N_TOTAL_NEURONS_2E × bool]
//
// 候选对编码 (重要):
//   CoactTracker.candidate_pre 为单 int 字段, 但需标识 (i, j) 两个神经元。
//   N_TOTAL_NEURONS_2E = 60000 < 2^16, 故用 16+16 位编码:
//       candidate_pre = i | (j << 16)      (规范化 i < j, 保证键唯一)
//   空槽位 candidate_pre == 0; 合法对最小键 (0,1) -> 65536 > 0, 不会与空槽冲突。
// =============================================================================

#include "config.h"
#include "types.h"
#include <cuda_runtime.h>

namespace stage2e {

// 共激活采样 (每步调用)
// 每 thread 处理一个候选采样对; 单 block + 动态 shared memory 收集发放神经元索引
//   d_trackers       [max_trackers] CoactTracker 数组
//   d_tracker_count  [1] 当前已用条目数 (atomic 推进, append-only)
//   d_spike_flags    [n_neurons] 当前步发放标志
//   current_da       当前 DA 浓度 (累加到 modulator_score)
//   sample_size      本步采样对数 (如 COACT_SAMPLE_SIZE = 500)
//   seed             随机种子 (建议混入 current_step 增加熵)
void launch_coactivation_sample(
    CoactTracker* d_trackers, int* d_tracker_count,
    const bool* d_spike_flags, float current_da,
    int n_neurons, int max_trackers, int sample_size,
    unsigned int seed, int current_step);

// 共激活淘汰 (周期性调用)
// 网格跨步遍历 tracker 数组; coact_count==0 且 (current_step - last_seen) >
// stale_threshold 的条目清零。append-only: 不递减 d_tracker_count, 仅清零槽位。
void launch_coactivation_prune(
    CoactTracker* d_trackers, int* d_tracker_count,
    int max_trackers, int current_step, int stale_threshold);

// P2 修复: 共激活计数衰减 (周期性调用)
// 对所有已填充 tracker 执行 coact_count *= COACT_DECAY_FACTOR, modulator_score 同步衰减
// 调用时机: 每 COACT_DECAY_INTERVAL 步, 在 launch_coactivation_sample 之后执行
// 效果: 低频共激活对自然归零被淘汰, 高频对维持 coact_count > form_threshold
void launch_coactivation_decay(
    CoactTracker* d_trackers, int* d_tracker_count, int max_trackers);

// =============================================================================
// Task 7: 结构可塑性批量重建
// =============================================================================
// 每 STRUCTURAL_REBUILD_INTERVAL 步触发:
//   1. structural_rebuild_kernel: 候选生成 (新突触) + 修剪标记 (弱突触)
//   2. host 读取计数, 判定总变更是否 > STRUCTURAL_CHANGE_THRESHOLD (5%)
//   3. 若超阈, launch_csr_rebuild 执行分块原地 CSR 重建
//
// 缓冲区 (由调用方/scheduler 分配):
//   - d_new_synapse_pairs   [COACT_MAX_NEW_SYNAPSES × 2] 新突触 (pre, post) 对
//   - d_new_synapse_count   [1] 新突触计数 (kernel atomic 推进)
//   - d_new_modulator_scores[COACT_MAX_NEW_SYNAPSES] 调质分数 (用于排序参考)
//   - d_prune_marks         [n_synapses] 修剪标记 (1=修剪, 0=保留)
//   - d_prune_count         [1] 修剪计数 (kernel atomic 推进)
// =============================================================================

// 结构重建 kernel: 单次 launch 完成阶段1 (候选生成) + 阶段2 (修剪标记)
//   阶段3 (5% 判定) 由 host wrapper 在 kernel 完成 + sync 后执行
//   d_trackers            [tracker_count] 共激活跟踪器
//   d_new_synapse_pairs   [max_new × 2] 输出新突触 (pre, post)
//   d_new_synapse_count   [1] 输出新突触计数 (调用前需清零)
//   d_new_modulator_scores[max_new] 输出调质分数
//   d_synapses / d_row_ptr 现有 CSR 突触 (row_ptr[n_neurons] = 总突触数)
//   d_prune_marks         [n_synapses] 输出修剪标记 (调用前需清零)
//   d_prune_count         [1] 输出修剪计数 (调用前需清零)
//   form_threshold        共激活形成阈值 (COACT_FORM_THRESHOLD = 5)
//   prune_weight_threshold 修剪权重阈值 (PRUNE_WEIGHT_THRESHOLD = 0.05f)
//   max_new               新突触上限 (COACT_MAX_NEW_SYNAPSES = 5000)
__global__ void structural_rebuild_kernel(
    const CoactTracker* __restrict__ d_trackers,
    int tracker_count,
    int* __restrict__ d_new_synapse_pairs,
    int* __restrict__ d_new_synapse_count,
    float* __restrict__ d_new_modulator_scores,
    const BioSynapse* __restrict__ d_synapses,
    const int* __restrict__ d_row_ptr,
    int n_neurons,
    int* __restrict__ d_prune_marks,
    int* __restrict__ d_prune_count,
    int form_threshold,
    float prune_weight_threshold,
    int max_new);

// CSR 重建 kernel: 单 block 分块原地重建 (正确性优先, 避免双缓冲 640MB)
//   Phase A: 前向分块迁移存活突触 (双 __syncthreads 保证每轮 "先读后写")
//   Phase B: 写入新突触到 [surviving_total, new_total) 区间 (与 Phase A 不重叠)
//   Phase C: 拷贝 d_new_row_ptr → d_row_ptr
//   d_remap_table 同时容纳 [0, n_old) 旧突触映射 + [n_old, n_old+new_count) 新突触目标
//   d_prune_marks 已编码进 remap_table (-1=修剪), kernel 内不再读取
__global__ void csr_rebuild_kernel(
    BioSynapse* __restrict__ d_synapses,
    int* __restrict__ d_row_ptr,
    const int* __restrict__ d_new_synapse_pairs,
    int new_count,
    const int* __restrict__ d_prune_marks,
    int n_neurons,
    int* __restrict__ d_new_row_ptr,
    int* __restrict__ d_remap_table);

// 结构重建 host wrapper (阶段1+2): 清零计数/标记 → launch kernel
// 注意: 不在此 sync, 调用方 (scheduler) 负责在读取计数前 sync
void launch_structural_rebuild(
    const CoactTracker* d_trackers, int tracker_count,
    int* d_new_pairs, int* d_new_count, float* d_new_scores,
    const BioSynapse* d_synapses, const int* d_row_ptr,
    int n_neurons, int* d_prune_marks, int* d_prune_count,
    int form_thr, float prune_thr, int max_new);

// CSR 重建 host wrapper (含 5% 判定 + CPU 端构建 row_ptr/remap_table + GPU 迁移)
//   返回 true  = 执行了重建 (变更 > 5% 阈值)
//   返回 false = 跳过重建 (变更 <= 5% 阈值)
//   临时缓冲 (d_new_row_ptr + d_remap_table ≈ 43MB) 内部分配, 重建后立即释放
//   d_new_row_ptr / d_remap_table 参数保留以匹配 kernel 签名, wrapper 内部自行分配
bool launch_csr_rebuild(
    BioSynapse* d_synapses, int* d_row_ptr,
    const int* d_new_pairs, int new_count,
    const int* d_prune_marks, int n_neurons,
    int n_synapses_total,
    int* d_new_row_ptr, int* d_remap_table,
    cudaStream_t stream);

// =============================================================================
// Task 19: CSR 完整性运行时校验
// =============================================================================
// csr_integrity_check_kernel: 重建后校验 CSR 数据结构合法性
//   1. row_ptr 单调性: row_ptr[i+1] >= row_ptr[i] 对所有 i ∈ [0, n_neurons)
//   2. col_ind 范围: 0 <= synapses[s].post_idx < n_neurons 对所有 s
//      (注: BioSynapse 中无独立 col_ind, 用 post_idx 替代, 语义一致)
//   3. 总数一致: row_ptr[n_neurons] == n_synapses_expected
//
// 错误码 (原子累积到 d_check_result):
//   bit0 (1): row_ptr 单调性失败
//   bit1 (2): col_ind 范围越界
//   bit2 (4): row_ptr[n_neurons] 与期望总数不一致
//   0 = 全部通过
//
// 校验通过 (d_check_result==0): 释放旧副本, 输出通过日志
// 校验失败 (d_check_result!=0): 从旧副本回滚, 输出错误日志 (含错误码)
// =============================================================================
__global__ void csr_integrity_check_kernel(
    const int* __restrict__ d_row_ptr,
    const BioSynapse* __restrict__ d_synapses,
    int n_neurons,
    int n_synapses_expected,
    int* __restrict__ d_check_result);

// CSR 完整性校验 host wrapper
//   1. 清零 d_check_result
//   2. 启动 kernel 校验
//   3. 同步并读取结果
//   返回值: 0=通过, !=0=错误码 (bit0/bit1/bit2)
int launch_csr_integrity_check(
    const int* d_row_ptr,
    const BioSynapse* d_synapses,
    int n_neurons,
    int n_synapses_expected,
    int* d_check_result,
    cudaStream_t stream);

// 带 CSR 完整性校验的重建 host wrapper (Task 19)
//   1. 保存旧 CSR 副本 (row_ptr + synapses)
//   2. 调用 launch_csr_rebuild 执行重建
//   3. 重建后调用 launch_csr_integrity_check 校验
//   4. 校验通过: 释放旧副本, 返回 0
//   5. 校验失败: 从旧副本回滚, 返回错误码
//   注: 若 CSR_INTEGRITY_CHECK_ENABLED=0, 等价于直接调用 launch_csr_rebuild
int launch_csr_rebuild_with_integrity_check(
    BioSynapse* d_synapses, int* d_row_ptr,
    const int* d_new_pairs, int new_count,
    const int* d_prune_marks, int n_neurons,
    int n_synapses_total,
    cudaStream_t stream);

} // namespace stage2e

#endif // SNN_STAGE2E_COACTIVATION_KERNELS_CUH
