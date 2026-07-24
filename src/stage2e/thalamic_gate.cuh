#ifndef SNN_STAGE2E_THALAMIC_GATE_CUH
#define SNN_STAGE2E_THALAMIC_GATE_CUH

// =============================================================================
// Stage 2e 丘脑-皮层门控 (§1.1 注意力门控)
// =============================================================================
// 每柱维护独立门控状态, 由活动水平 + novelty 驱动, 动态调制输入增益
// 门控状态独立存储, 不嵌入 NeuronStateAdEx (保持 56B 对齐)
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include <cstdint>

namespace stage2e {

// 丘脑门控状态 (16B, 每柱一个)
// gate_signal ∈ [0,1]: 0=完全闭门, 1=完全开门
// activity_ema: 慢速活动估计(spike count 滑动平均)
// novelty_ema: novelty 慢速估计
// l6_activity_ema: L6 层活动慢速估计 (Phase R2 模块 C: 皮层-丘脑闭环反馈)
struct ThalamicGateState {
    float gate_signal;      // 4B  门控信号 ∈ [0, 1]
    float activity_ema;     // 4B  活动滑动平均
    float novelty_ema;      // 4B  novelty 滑动平均
    float l6_activity_ema;  // 4B  L6 活动滑动平均 (替代 _pad, 保持 16B 对齐)
};
static_assert(sizeof(ThalamicGateState) == 16, "ThalamicGateState must be 16 bytes");

// 初始化所有柱的门控状态 (gate_signal=GATE_INITIAL_SIGNAL, ema=0)
// d_gate_states: 设备指针, 大小 = N_COLUMNS_2E * sizeof(ThalamicGateState)
void init_thalamic_gate(ThalamicGateState* d_gate_states);

// 每步更新门控状态 (活动补偿 + novelty 增强 + L6 反馈闭环)
// d_gate_states: 门控状态数组
// d_column_spikes: 每柱当前 spike count (设备指针, 全柱统计)
// d_l6_column_spikes: 每柱 L6 层 spike count (设备指针, 仅 region==3 神经元)
// current_byte: 当前注入字节 (非注入步传 0)
// is_inject_step: 是否为注入步
// d_byte_history: 字节历史计数数组 (256 个 uint32), 用于 novelty 计算
// d_gate_stats: 统计输出 [gate_mean, gate_open_ratio, gate_min, gate_max] (float[4])
void launch_thalamic_gate_update(ThalamicGateState* d_gate_states,
                                  const int* d_column_spikes,
                                  const int* d_l6_column_spikes,
                                  uint8_t current_byte,
                                  bool is_inject_step,
                                  unsigned int* d_byte_history,
                                  float* d_gate_stats);

} // namespace stage2e

#endif // SNN_STAGE2E_THALAMIC_GATE_CUH
