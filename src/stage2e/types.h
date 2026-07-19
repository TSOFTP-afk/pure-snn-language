#ifndef SNN_STAGE2E_TYPES_H
#define SNN_STAGE2E_TYPES_H

// =============================================================================
// Stage 2e 数据类型定义 (v4)
// =============================================================================
// 对应设计文档 §2:
//   - NeuronStateAdEx: 56B, AdEx 神经元状态 (v3)
//   - BioSynapse: 80B, 含突触级调质受体密度 (v3 强化 B)
//   - HippoIndex: 256B, 海马索引 (v3 强化 C, 50 维签名)
//   - CoactTracker: 16B, 共激活跟踪 (v3 强化 D)
//   - WMSlot: 216B, 工作记忆槽位 (v3 强化 E)
// =============================================================================

#include "config.h"
#include <cstdint>

// 当被纯 C++ 文件（main.cpp 等）包含时，__host__/__device__ 关键字未定义
// 需要 fallback 为空宏，让 MSVC 也能编译
#ifndef __host__
  #define __host__
#endif
#ifndef __device__
  #define __device__
#endif

// -----------------------------------------------------------------------------
// AdEx 神经元状态 (56B, 16 字节对齐)
// v3: 替代 stage0 的 24B NeuronState
// -----------------------------------------------------------------------------
struct NeuronStateAdEx {
    // 基础电生理 (20B)
    float membrane_potential;       // 4B  offset 0   归一化电压 V_norm ∈ [-0.5, 1.5]
    float synaptic_current;         // 4B  offset 4   突触输入电流 (AMPA + GABA)
    float nmda_current;             // 4B  offset 8   NMDA 电流 (电压依赖)
    float adaptive_conductance;     // 4B  offset 12  AdEx 适应电导 w
    int   last_spike_time;          // 4B  offset 16  上次脉冲时间步

    // 放电统计 (12B)
    float fire_rate;                // 4B  offset 20  滑动平均发放率
    int   refractory_remaining;     // 4B  offset 24  不应期剩余步数
    float ca_neuron;                // 4B  offset 28  神经元级钙浓度

    // 类型与分区 (8B)
    uint8_t  neuron_type;           // 1B  offset 32  0=兴奋, 1=抑制
    uint8_t  region;                // 1B  offset 33  0=sensory, 1=assoc, 2=motor, 3=prefrontal
    uint8_t  inhibitory_subtype;    // 1B  offset 34  InhibitorySubtype (FS/LTS/SOM)
    uint8_t  column_id;             // 1B  offset 35  所属柱 (0..49) 或 255=前额叶
    int16_t  pf_group_id;           // 2B  offset 36  前额叶组 ID (-1=非前额叶)
    uint16_t threshold_offset;      // 2B  offset 38  定点阈值偏移 (实际 = /256.0f)

    // WM 注入 + 缩放 (8B)
    float wm_injection;             // 4B  offset 40  工作记忆注入电流
    float homeostatic_factor;       // 4B  offset 44  局部突触缩放因子

    // padding (8B) 到 56B
    int   _reserved;                // 4B  offset 48
    int   _pad;                     // 4B  offset 52
};
static_assert(sizeof(NeuronStateAdEx) == 56,
              "NeuronStateAdEx must be 56 bytes (v3 spec)");

// -----------------------------------------------------------------------------
// BioSynapse: 严格 80B (v3 强化 B + v4 强化 I/J/K)
// -----------------------------------------------------------------------------
struct BioSynapse {
    // 0-15: 基础连接 (16B)
    int   pre_idx;                  // 4B  offset 0   突触前神经元索引
    int   post_idx;                 // 4B  offset 4   突触后神经元索引
    float weight;                   // 4B  offset 8   突触权重 (兴奋>0, 抑制<0)
    float delay_steps;              // 4B  offset 12  v4 强化 I: 传导延迟步数

    // 16-31: STDP 双 trace (16B, v4 强化 J)
    float last_pre_spike;           // 4B  offset 16
    float last_post_spike;          // 4B  offset 20
    float x_pre_trace;              // 4B  offset 24  v4: 突触前 trace
    float x_post_trace;             // 4B  offset 28  v4: 突触后 trace

    // 32-47: 电导 + 钙 (16B)
    float nmda_conductance;         // 4B  offset 32  NMDA 电导
    float ampa_conductance;         // 4B  offset 36  AMPA 电导
    float ca_concentration;         // 4B  offset 40  钙浓度 (LTP/LTD 开关)
    float resource;                 // 4B  offset 44  STP 资源 R

    // 48-63: Eligibility + 缩放 (16B)
    float eligibility;              // 4B  offset 48  1 阶 eligibility trace
    float eligibility_slow;         // 4B  offset 52  v3 强化 H: 2 阶慢 trace
    float utilization;              // 4B  offset 56  STP 利用率 U
    float scaling_factor;           // 4B  offset 60  局部突触缩放因子

    // 64-79: CaMKII + 调质受体 (16B, v3 强化 B + v4 强化 K)
    float camkii_autophosph;        // 4B  offset 64  v4 强化 K: 自磷酸化水平 [0,1]
    float da_receptor;              // 4B  offset 68  DA 受体密度 (D1+/D2-)
    float ach_receptor;             // 4B  offset 72  ACh 受体密度
    uint8_t receptor_flags;         // 1B  offset 76  AMPA|NMDA|GABA_A|GABA_B
    uint8_t ne_receptor_u8;         // 1B  offset 77  NE 受体 (定点: /127.0f)
    uint8_t ht5_receptor_u8;        // 1B  offset 78  5HT 受体 (定点: /127.0f)
    uint8_t _pad;                   // 1B  offset 79
};
static_assert(sizeof(BioSynapse) == 80,
              "BioSynapse must be exactly 80 bytes (v3 强化 B + v4 强化 I/J/K)");

// 内联辅助: 获取 NE/5HT 受体密度 (从定点恢复)
__host__ __device__ inline float get_ne_receptor(const BioSynapse& s) {
    return static_cast<float>(s.ne_receptor_u8) / 127.0f;
}
__host__ __device__ inline float get_ht5_receptor(const BioSynapse& s) {
    return static_cast<float>(s.ht5_receptor_u8) / 127.0f;
}
__host__ __device__ inline void set_ne_receptor(BioSynapse& s, float v) {
    s.ne_receptor_u8 = static_cast<uint8_t>(v * 127.0f);
}
__host__ __device__ inline void set_ht5_receptor(BioSynapse& s, float v) {
    s.ht5_receptor_u8 = static_cast<uint8_t>(v * 127.0f);
}

// -----------------------------------------------------------------------------
// 海马体索引 (256B, v3 强化 C: 50K × 256B = 12.8MB)
// -----------------------------------------------------------------------------
struct HippoIndex {
    float pattern_signature[50];    // 200B  v3: 50 维 PCA 签名
    int   pattern_start_step;       // 4B
    int   replay_count;             // 4B
    float importance;               // 4B
    float _pad[11];                 // 44B  凑齐 256B (16 字节对齐)
};
static_assert(sizeof(HippoIndex) == 256,
              "HippoIndex must be 256 bytes (v3 强化 C)");

// -----------------------------------------------------------------------------
// 共激活跟踪器 (16B, v3 强化 D: 500K × 16B = 8MB)
// -----------------------------------------------------------------------------
struct CoactTracker {
    int   candidate_pre;            // 4B  候选前突触神经元
    int   coact_count;              // 4B  共激活计数
    int   last_seen;                // 4B  上次更新时间步
    float modulator_score;          // 4B  v3: 调质加权分数
};
static_assert(sizeof(CoactTracker) == 16,
              "CoactTracker must be 16 bytes (v3 强化 D)");

// -----------------------------------------------------------------------------
// 工作记忆槽位 (216B, v3 强化 E: 50 槽)
// -----------------------------------------------------------------------------
struct WMSlot {
    float pattern[50];              // 200B  v3: 50 维 PCA 签名
    int   age;                      // 4B
    float activation;               // 4B
    int   prefrontal_group;         // 4B    v3: 绑定的前额叶组 ID
    int   _pad;                     // 4B    凑齐 216B
};
static_assert(sizeof(WMSlot) == 216,
              "WMSlot must be 216 bytes (v3 强化 E)");

// -----------------------------------------------------------------------------
// 发育阶段 (v2 修复 5)
// -----------------------------------------------------------------------------
enum class DevPhase : uint8_t {
    EMBRYONIC    = 0,   // 0 - 30K
    SYNAPTOGENIC = 1,   // 30K - 200K
    CRITICAL     = 2,   // 200K - 800K
    PRUNING      = 3,   // 800K - 1.5M
    MATURE       = 4,   // 1.5M - 3M
};

struct DevPhaseParams {
    float plasticity_gain;          // STDP 学习率乘数
    float prune_threshold;          // 修剪阈值
    float growth_rate;              // 新突触生成率
    float nmda_expression;          // NMDA 受体表达量
    float ach_level;                // 乙酰胆碱基线
    float myeline_factor;           // 传导速度乘数
    int   end_step;                 // 阶段结束步数
};

// -----------------------------------------------------------------------------
// 网络统计 (用于监控)
// -----------------------------------------------------------------------------
struct NetworkStats2e {
    int   total_spikes;
    int   excitatory_spikes;
    int   inhibitory_spikes;
    float mean_fire_rate;
    float mean_weight;
    float mean_camkii;              // v4: 平均 CaMKII 自磷酸化水平
    float da_level;
    float ach_level;
    float ne_level;
    float ht5_level;
    float mean_nmda_conductance;
    int   active_synapses;
    int   pruned_this_step;
    int   formed_this_step;
    size_t vram_used_bytes;
    size_t vram_peak_bytes;
};

#endif // SNN_STAGE2E_TYPES_H
