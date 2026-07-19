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
// Stage 2 (方案 A)：柱内三层流水线布局。
//   每柱 1000 神经元 = 200 sensory + 600 association + 200 motor
//   柱 c 的 sensory 层:    [c*1000,       c*1000 + 200)
//   柱 c 的 association 层: [c*1000 + 200, c*1000 + 800)
//   柱 c 的 motor 层:       [c*1000 + 800, c*1000 + 1000)
//
// Stage 0 默认仍是全局脑区划分（sensory 在前，association 中间，motor 末尾）。
// 用 #ifdef SNN_STAGE2_BUILD 切换，stage0 编译路径完全不变。
// =============================================================================
#ifdef SNN_STAGE2_BUILD
  // Stage 2 方案 A: 柱内分层
  // 这两个常量必须与 stage2/config.h 的 COL_SENSORY_SIZE / COL_ASSOCIATION_SIZE 一致
  #ifndef STAGE2_COL_SENSORY_SIZE
    #define STAGE2_COL_SENSORY_SIZE      200
  #endif
  #ifndef STAGE2_COL_ASSOCIATION_SIZE
    #define STAGE2_COL_ASSOCIATION_SIZE  600
  #endif
  #ifndef STAGE2_NEURONS_PER_COLUMN
    #define STAGE2_NEURONS_PER_COLUMN    1000
  #endif

  __host__ __device__ inline int region_start(BrainRegion region) {
    // 方案 A 下脑区不再是连续段，region_start 失去意义。
    // 返回 0 保持接口兼容（stage0 kernel 只在划分逻辑里调用，且 stage2 编译时
    // 不会用 region_start 来定位——改用 get_region + 柱内偏移）。
    (void)region;
    return 0;
  }

  __host__ __device__ inline int region_size(BrainRegion region) {
    // 方案 A: 每柱内 sensory=200, association=600, motor=200
    // 全局总数仍是 2000/6000/2000，与 stage0 一致
    switch (region) {
        case BrainRegion::SENSORY:     return N_SENSORY_NEURONS;     // 2000
        case BrainRegion::ASSOCIATION: return N_ASSOCIATION_NEURONS; // 6000
        case BrainRegion::MOTOR:       return N_MOTOR_NEURONS;       // 2000
        default:                       return 0;
    }
  }

  // 方案 A: 根据柱内偏移判断脑区
  __host__ __device__ inline BrainRegion get_region(int idx) {
      int off = idx % STAGE2_NEURONS_PER_COLUMN;  // 柱内偏移 0..999
      if (off < STAGE2_COL_SENSORY_SIZE) {
          return BrainRegion::SENSORY;
      } else if (off < STAGE2_COL_SENSORY_SIZE + STAGE2_COL_ASSOCIATION_SIZE) {
          return BrainRegion::ASSOCIATION;
      } else {
          return BrainRegion::MOTOR;
      }
  }
#else
  // Stage 0 默认: 全局脑区划分
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
#endif

// 根据神经元全局索引判断其类型（每脑区内 80/20 划分兴奋/抑制）
// 必须与 neuron_kernel.cu 的划分逻辑保持一致
__host__ __device__ inline NeuronType get_neuron_type(int idx) {
#ifdef SNN_STAGE2_BUILD
    // 方案 A: 柱内分层
    //   柱内 sensory 层:  前 80% (160 个) 兴奋, 后 20% (40 个) 抑制
    //   柱内 association 层: 前 80% (480 个) 兴奋, 后 20% (120 个) 抑制
    //   柱内 motor 层:    前 80% (160 个) 兴奋, 后 20% (40 个) 抑制
    // 这样每柱 800 兴奋 + 200 抑制 = 80/20 全局比例保持
    int off = idx % STAGE2_NEURONS_PER_COLUMN;  // 0..999
    int layer_off;  // 层内偏移
    int layer_size;
    if (off < STAGE2_COL_SENSORY_SIZE) {
        layer_off = off;
        layer_size = STAGE2_COL_SENSORY_SIZE;
    } else if (off < STAGE2_COL_SENSORY_SIZE + STAGE2_COL_ASSOCIATION_SIZE) {
        layer_off = off - STAGE2_COL_SENSORY_SIZE;
        layer_size = STAGE2_COL_ASSOCIATION_SIZE;
    } else {
        layer_off = off - STAGE2_COL_SENSORY_SIZE - STAGE2_COL_ASSOCIATION_SIZE;
        layer_size = STAGE2_NEURONS_PER_COLUMN - STAGE2_COL_SENSORY_SIZE - STAGE2_COL_ASSOCIATION_SIZE;
    }
    int layer_exc_count = (int)(layer_size * EXCITATORY_RATIO);
    return (layer_off < layer_exc_count) ? NeuronType::EXCITATORY
                                          : NeuronType::INHIBITORY;
#else
    BrainRegion region = get_region(idx);
    int rs = region_start(region);
    int rn = region_size(region);
    int region_idx = idx - rs;
    int region_exc_count = (int)(rn * EXCITATORY_RATIO);
    return (region_idx < region_exc_count) ? NeuronType::EXCITATORY
                                           : NeuronType::INHIBITORY;
#endif
}

#endif // SNN_TYPES_H
