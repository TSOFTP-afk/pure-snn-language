#ifndef SNN_TYPES_H
#define SNN_TYPES_H

#include "config.h"

// =============================================================================
// 数据类型定义
// =============================================================================

// 神经元类型
enum class NeuronType : unsigned char {
    EXCITATORY = 0,   // 兴奋性（AMPA 型）
    INHIBITORY = 1,   // 抑制性（GABA 型）
};

// 脑区类型
enum class BrainRegion : unsigned char {
    SENSORY     = 0,   // 感觉皮层
    ASSOCIATION = 1,   // 联合皮层
    MOTOR       = 2,   // 运动皮层
};

// =============================================================================
// 神经元状态（每个神经元一份，存储在 GPU 全局内存）
// 字段排列：4 bytes 类型在前，1 byte 类型在后，减少 padding
// =============================================================================
struct NeuronState {
    float membrane_potential;     // 4 bytes, offset 0
    float synaptic_current;       // 4 bytes, offset 4
    int   last_spike_time;        // 4 bytes, offset 8
    int   refractory_remaining;   // 4 bytes, offset 12
    float fire_rate;              // 4 bytes, offset 16
    NeuronType type;              // 1 byte,  offset 20
    BrainRegion region;           // 1 byte,  offset 21
    signed char threshold_offset; // 1 byte,  offset 22, 定点：实际值 = stored / 16.0f
    unsigned char _pad;           // 1 byte,  offset 23
};

// 大小断言：确保结构体大小符合预期（用于显存估算）
static_assert(sizeof(NeuronState) == 24,
              "NeuronState size mismatch, check field alignment");

// =============================================================================
// 突触状态（每个突触一份，存储在 GPU 全局内存）
// =============================================================================
struct Synapse {
    int   pre_idx;                // 突触前神经元索引
    int   post_idx;               // 突触后神经元索引
    float weight;                 // 突触强度
    float delay;                  // 突触延迟（时间步）
    float last_pre_spike;         // 上次突触前脉冲时间（用于 STDP）
    float last_post_spike;        // 上次突触后脉冲时间（用于 STDP）
    float eligibility;            // Eligibility trace（用于延迟 reward 归因）
    float _pad;                   // 对齐到 32 bytes
};

static_assert(sizeof(Synapse) == 32,
              "Synapse size mismatch");

// =============================================================================
// 调质系统状态（全局，少量）
// =============================================================================
struct NeuromodulatorState {
    float dopamine;               // 多巴胺水平（奖励调制 STDP）
    float serotonin;              // 血清素水平（探索调制）
    float energy;                 // 内感受：能量
    float arousal;                // 内感受：唤醒度
};

// =============================================================================
// 网络统计（用于监控）
// =============================================================================
struct NetworkStats {
    int   total_spikes;           // 当前时间步总脉冲数
    int   excitatory_spikes;
    int   inhibitory_spikes;
    float mean_fire_rate;         // 平均发放率
    float mean_weight;            // 平均突触权重
    int   active_synapses;        // 活跃突触数
    float dopamine_level;
    float serotonin_level;
};

// =============================================================================
// 辅助函数：脑区偏移
// =============================================================================
__host__ __device__ inline int region_start(BrainRegion region) {
    switch (region) {
        case BrainRegion::SENSORY:     return 0;
        case BrainRegion::ASSOCIATION: return N_SENSORY_NEURONS;
        case BrainRegion::MOTOR:       return N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS;
        default:                       return 0;
    }
}

__host__ __device__ inline int region_size(BrainRegion region) {
    switch (region) {
        case BrainRegion::SENSORY:     return N_SENSORY_NEURONS;
        case BrainRegion::ASSOCIATION: return N_ASSOCIATION_NEURONS;
        case BrainRegion::MOTOR:       return N_MOTOR_NEURONS;
        default:                       return 0;
    }
}

// 根据神经元全局索引判断其所在脑区
__host__ __device__ inline BrainRegion get_region(int idx) {
    if (idx < N_SENSORY_NEURONS) {
        return BrainRegion::SENSORY;
    } else if (idx < N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS) {
        return BrainRegion::ASSOCIATION;
    } else {
        return BrainRegion::MOTOR;
    }
}

// 根据神经元全局索引判断其类型（每脑区内 80/20 划分兴奋/抑制）
// 必须与 neuron_kernel.cu 的划分逻辑保持一致
__host__ __device__ inline NeuronType get_neuron_type(int idx) {
    BrainRegion region = get_region(idx);
    int rs = region_start(region);
    int rn = region_size(region);
    int region_idx = idx - rs;
    int region_exc_count = (int)(rn * EXCITATORY_RATIO);
    return (region_idx < region_exc_count) ? NeuronType::EXCITATORY
                                           : NeuronType::INHIBITORY;
}

#endif // SNN_TYPES_H
