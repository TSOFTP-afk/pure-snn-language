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
// 调质浓度已分配: d_da/ach/ne/ht5_concentration (55K × 4B)
// 价值函数已分配: d_w_value (200), d_w_pred (200×200), d_pred_fr (200),
//                  d_subcolumn_fr (200), d_baseline_fr (200)
// 字节直方图: d_byte_histogram (256)
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

namespace stage2e {

// 调质浓度动力学 (每100步)
// 输入: d_da/ach/ne/ht5_concentration (当前浓度)
//       reward_signal, novelty, pred_succ, kl_divergence (host 传入)
// 输出: 更新后的调质浓度
//
// DA: δ(t) = R(t) + γ·V(s') - V(s); DA半衰期5步
// ACh: 基线0.2, 惊奇+Δ, 注意力+Δ
// NE: 基线0.05, KL散度触发脉冲
// 5HT: 基线0.1, 预测误差持续负时上升
void launch_modulatory(MemoryAllocator* alloc, int step,
                       float reward_signal, float novelty,
                       float pred_succ, float kl_divergence,
                       float da_delta);

// DA价值函数更新 (每100步)
// 输入: d_subcolumn_fr (当前亚柱发放直方图), d_baseline_fr (EMA基线)
//       d_w_value, d_w_pred (学习参数)
// 输出: 更新后的 w_value, w_pred, baseline_fr, pred_fr
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

ModulatoryStats get_modulatory_stats(MemoryAllocator* alloc);

} // namespace stage2e

#endif // SNN_STAGE2E_MODULATORY_KERNELS_CUH
