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

struct DelayQueueRuntimeStats {
    long long arrived_events;
    long long dispatched_events;
    long long dropped_events;
    int last_arrived_events;
    int last_dispatched_events;
    int last_dropped_events;
    int max_slot_depth;
};

struct DelayQueueCheckpointState {
    int ring_counter_history[DELAY_STEPS_MAX];
    DelayQueueRuntimeStats stats;
};

// 把上一轮写入当前 ring_idx 槽位的延迟电流注入 input_current
// 内部: 清零 input_current, 启动 delay_inject_kernel, 消费后清空当前槽位计数
void launch_delay_inject(MemoryAllocator* alloc, int ring_idx);

// AdEx 神经元更新 (§2.1)
// 输入: input_current (含延迟注入 + 外部输入), nmda_current, inhibitory_current
// 输出: neurons (V, w, refractory, fire_rate, threshold_offset), spike_flags
void launch_lif_adex(MemoryAllocator* alloc, int step, const struct DevPhaseParams& phase,
                     int* d_single_neuron_burst_counter);

// 按突触 delay_steps 把 pre 脉冲分发到环形队列
// launch 只入队 GPU 工作；finish 与本步其他统计共用一次 host 同步。
void launch_delay_dispatch(MemoryAllocator* alloc, int step, int ring_idx);
void finish_delay_dispatch();

int delay_queue_last_arrived_events();
const DelayQueueRuntimeStats& delay_queue_stats();
bool export_delay_queue_state(DelayQueueCheckpointState* state);
bool import_delay_queue_state(const DelayQueueCheckpointState& state);

// ==================== Task 7: 运动皮层 AdEx 更新 ====================
//
// 复用现有 lif_adex_kernel 对运动皮层神经元 (d_motor_neurons) 进行 AdEx 更新。
//   - 输入电流: 来自 launch_l5_to_motor_synapse 写入的 motor_input_current
//   - NMDA / 抑制电流: 简化为零缓冲 (运动皮层暂不实现 NMDA/抑制)
//   - burst 计数: 内部静态计数器 (与主网络 single_neuron_burst_counter 分离,
//     不暴露给 scheduler, 避免新增 checkpoint 字段)
//
// 缓冲区: 零缓冲 (nmda/inhibitory/burst_counter) 为静态 device 缓冲, 懒分配, 生命周期 = 程序
void launch_motor_adex(MemoryAllocator* alloc, int step,
                       const struct DevPhaseParams& phase);

} // namespace stage2e

#endif // SNN_STAGE2E_NEURON_KERNELS_CUH
