#ifndef SNN_STAGE2E_NEURON_KERNELS_CUH
#define SNN_STAGE2E_NEURON_KERNELS_CUH

// =============================================================================
// Stage 2e 神经元相关 kernel (P1)
// =============================================================================
// 对应设计文档 §2.1, §2.2 v4 强化 I:
//   - delay_inject:   把上一轮写入当前 ring_idx 槽位的延迟电流注入 input_current
//   - lif_adex:       AdEx 两室神经元 (适应性 + 簇状发放 + 阈值动态)
//   - delay_dispatch: 按突触 delay_steps 把 pre 脉冲分发到环形队列 (写下一轮槽位)
//
// 调用顺序 (每步):
//   1. launch_delay_inject(alloc, ring_idx)
//   2. launch_input_inject(...)  (外部群体编码输入, 在 input_encoding.cu)
//   3. launch_lif_adex(alloc, step, phase)
//   4. launch_synapse_nmda(...)  (在 synapse_kernels.cu)
//   5. launch_stdp_dual_trace(...)
//   6. launch_stdp_stp(...)
//   7. launch_delay_dispatch(alloc, step, ring_idx)
//   8. ring_idx = (ring_idx + 1) % DELAY_STEPS_MAX
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

namespace stage2e {

// 延迟环形队列每槽位最大活跃突触数 (估计每步 ~500K 活跃)
#define DELAY_RING_SLOT_CAPACITY 500000

// 把上一轮写入当前 ring_idx 槽位的延迟电流注入 input_current
// 内部: 清零 input_current, 启动 delay_inject_kernel
void launch_delay_inject(MemoryAllocator* alloc, int ring_idx);

// AdEx 神经元更新 (§2.1)
// 输入: input_current (含延迟注入 + 外部输入), nmda_current, inhibitory_current
// 输出: neurons (V, w, refractory, fire_rate, threshold_offset), spike_flags
void launch_lif_adex(MemoryAllocator* alloc, int step, const struct DevPhaseParams& phase);

// 按突触 delay_steps 把 pre 脉冲分发到环形队列
// 内部: 清零计数器, 启动 delay_dispatch_kernel, 同步, 拷贝计数器回 host
void launch_delay_dispatch(MemoryAllocator* alloc, int step, int ring_idx);

} // namespace stage2e

#endif // SNN_STAGE2E_NEURON_KERNELS_CUH
