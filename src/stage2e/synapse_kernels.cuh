#ifndef SNN_STAGE2E_SYNAPSE_KERNELS_CUH
#define SNN_STAGE2E_SYNAPSE_KERNELS_CUH

// =============================================================================
// Stage 2e 突触相关 kernel (P1)
// =============================================================================
// 对应设计文档 §2.2:
//   - synapse_nmda_kernel: NMDA 受体电压依赖 (Mg²⁺ 阻塞) + 钙浓度更新
//   - stdp_dual_trace_kernel: STDP 双 trace (x_pre/x_post) + Δw 计算
//   - stdp_stp_kernel: 短期可塑性 (易化/抑制)
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

namespace stage2e {

// NMDA 受体电压依赖 + 钙浓度更新 (§2.2)
// 输入: d_synapses, d_neurons (post V), d_spike_flags (pre 是否脉冲)
// 输出: d_synapses (nmda_conductance, ampa_conductance, ca_concentration)
//       d_nmda_current (累积到 post 神经元的 NMDA 电流)
//
// 数学:
//   V_bio = -70 + 50 * V_norm                       (电压重映射)
//   g_NMDA(V) = g_max / (1 + [Mg²⁺] * exp(-V_bio/16.13) / 3.57)
//   当 V_bio < -60mV: NMDA 闭合 (Mg²⁺ 阻塞)
//   当 V_bio > -20mV: NMDA 开放
//
// 每 thread 处理一个突触
void launch_synapse_nmda(MemoryAllocator* alloc, int step, int arrived_ring_idx, int arrived_count);

// STDP 双 trace (v4 强化 J, Bi & Poo 2001)
// 输入: d_synapses (last_pre/post_spike), d_spike_flags, d_stdp_x_pre_trace
// 输出: d_synapses (x_pre_trace, x_post_trace, weight, last_pre/post_spike)
//
// 数学:
//   x_pre(t+1)  = x_pre(t)  * exp(-dt/τ_pre)  + A_plus  * pre_spike
//   x_post(t+1) = x_post(t) * exp(-dt/τ_post) + A_minus * post_spike
//   Δw = +x_pre  * post_spike    (LTP: pre 在 post 之前)
//        -x_post * pre_spike     (LTD: post 在 pre 之前)
//   weight += η * Δw * plasticity_gain
//   兴奋性: weight clamp 到 [0, W_MAX]
//   抑制性: weight clamp 到 [-W_MAX, 0]
void launch_stdp_dual_trace(MemoryAllocator* alloc, int step, float plasticity_gain,
                            int arrived_ring_idx, int arrived_count);

// Checkpoints keep the historical AoS representation. Materialize lazy
// traces before saving, and reset transient epochs after loading.
void materialize_stdp_traces(MemoryAllocator* alloc, int step);
void reset_stdp_trace_epochs(MemoryAllocator* alloc, int step);

// 短期可塑性 STP (§2.2, Tsodyks-Markram 1998)
// 输入: d_synapses (resource, utilization), d_spike_flags (pre)
// 输出: d_synapses (resource, utilization)
//
// 数学:
//   pre 脉冲时:
//     u_new = U + u * (1 - U) * exp(-dt/τ_fac)   (易化)
//     r_new = r * (1 - u_new) * exp(-dt/τ_rec)   (抑制)
//     resource = r_new
//     utilization = u_new
void launch_stdp_stp(MemoryAllocator* alloc, int step, int arrived_ring_idx, int arrived_count);

// ==================== P2: 中时间尺度学习规则 ====================

// CaMKII 自磷酸化动力学 (§3, 每10步, Graupner & Brunel 2012)
// 输入: d_synapses (ca_concentration, camkii_autophosph), d_camkii_activity
// 输出: d_camkii_activity, d_synapses.camkii_autophosph
//
// d(activity)/dt = +k1·Ca^4·(1-activity) - k2·activity·PP1
// d(autophosph)/dt = +k3·activity^2·(1-autophosph) - k4·autophosph·PP1
// plasticity_factor = 1.0 - 0.5·autophosph
void launch_camkii(MemoryAllocator* alloc, int step);

// 2阶 eligibility trace (§3, 每10步)
// e1(t) = λ1·e1(t-1) + STDP_delta(t)  快 τ~20ms
// e2(t) = λ2·e2(t-1) + e1(t)          慢 τ~200ms
void launch_stdp_eligibility(MemoryAllocator* alloc, int step);

// 局部突触缩放 (§3.3, 每100步)
// scale_local(i) = (target_fr / mean_FR(i))^α
// w_ij *= scale_i · clamp(scale_j / scale_i, 0.5, 2.0)
void launch_synaptic_scaling(MemoryAllocator* alloc, int step, float target_fr);

// E0 消融模式: 设置 device 开关 (关闭三因素调制 + CaMKII)
void set_e0_ablation(bool enable);

} // namespace stage2e

#endif // SNN_STAGE2E_SYNAPSE_KERNELS_CUH
