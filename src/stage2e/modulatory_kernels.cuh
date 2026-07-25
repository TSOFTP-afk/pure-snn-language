#ifndef SNN_STAGE2E_MODULATORY_KERNELS_CUH
#define SNN_STAGE2E_MODULATORY_KERNELS_CUH

// =============================================================================
// Stage 2e 调质系统 + DA价值函数 + 字节选择性统计 (P2)
// =============================================================================
// 对应设计文档 §3.1-§3.2:
//   - modulatory_kernel: DA/ACh/NE/5HT 浓度动力学 (每100步)
//   - da_value_function: TD error + novelty + pred_succ (每100步)
//   - byte_histogram_kernel: 字节选择性统计 (每注入步)
//
// Task 10: W_pred 升级为完整 200×200 矩阵
//   - 预测: pred_fr = W_pred · fr_prev (完整矩阵-向量乘法, 替代对角项标量乘法)
//   - 更新: W_pred += η_pred · (fr - pred) ⊗ fr_prev (外积更新, 替代对角项更新)
//   - 预测成功率: (cos(pred_fr, fr_subcol) + 1) / 2 (余弦相似度, 替代二元判断)
//   - 新增缓冲: d_subcol_fr_prev (200, 上一步亚柱发放率)
//
// 调质浓度已分配: d_da/ach/ne/ht5_concentration (55K × 4B)
// 价值函数已分配: d_w_value (200), d_w_pred (200×200), d_pred_fr (200),
//                  d_subcolumn_fr (200), d_baseline_fr (200), d_subcol_fr_prev (200)
// 字节直方图: d_byte_histogram (256)
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

namespace stage2e {

// 调质浓度动力学 (每100步)
// 输入: d_da/ach/ne/ht5_concentration (当前浓度)
//       reward_signal, novelty, pred_succ, kl_divergence (host 传入)
//       prediction_error_norm: 在线解码预测误差 L2 范数 (||error||, ∈ [0, ~1.414])
// 输出: 更新后的调质浓度
//
// DA: δ(t) = R(t) + γ·V(s') - V(s); DA半衰期5步
//    预测误差耦合: DA = DA_BASE + DA_GAIN × (1 - ||error||) + TD 驱动
//    预测越准 (||error|| 越小) → DA 越高 (奖励预测准确性)
// ACh: 基线0.2, 惊奇+Δ, 注意力+Δ
// NE: 基线0.05, KL散度触发脉冲
// 5HT: 基线0.1, 预测误差持续负时上升
//
// DA 释放区域: [0, N_TOTAL_NEURONS_2E) = [0, 60000), 包含联合皮层 + 前额叶 + 运动皮层
void launch_modulatory(MemoryAllocator* alloc, int step,
                       float reward_signal, float novelty,
                       float pred_succ, float kl_divergence,
                       float da_delta,
                       float prediction_error_norm);

// DA价值函数更新 (每100步)
// Task 10: W_pred 完整矩阵 + 余弦相似度预测成功率
// 输入: d_subcolumn_fr (当前亚柱发放直方图), d_baseline_fr (EMA基线)
//       d_w_value, d_w_pred (学习参数), d_subcol_fr_prev (上一步 fr)
// 输出: 更新后的 w_value, w_pred (完整矩阵), baseline_fr, pred_fr, d_subcol_fr_prev
//       内部计算 h_pred_succ_cos (余弦相似度预测成功率, 供 launch_modulatory 使用)
//       返回 V(s) 和 V(s') 用于 TD error
void launch_da_value_function(MemoryAllocator* alloc, int step,
                              float reward, float* out_v_s, float* out_v_sp);

// 字节选择性直方图更新 (每注入步)
// 输入: d_spike_flags, 当前字节
// 输出: d_byte_histogram (256 bins, 累积每个字节的总 spike 数)
void launch_byte_histogram(MemoryAllocator* alloc, uint8_t current_byte);

// 获取字节直方图 (host 端拷贝)
void get_byte_histogram(MemoryAllocator* alloc, int* out_hist);

// 调质系统统计 (host 端读取)
struct ModulatoryStats {
    float da_mean;
    float ach_mean;
    float ne_mean;
    float ht5_mean;
    float v_s;
    float v_sp;
    float da_delta;
    float novelty;
    float pred_succ;
};

struct ModulatoryRuntimeState {
    float v_s;
    float v_sp;
};

ModulatoryStats get_modulatory_stats(MemoryAllocator* alloc);
ModulatoryRuntimeState export_modulatory_runtime_state();
void import_modulatory_runtime_state(const ModulatoryRuntimeState& state);

} // namespace stage2e

#endif // SNN_STAGE2E_MODULATORY_KERNELS_CUH
