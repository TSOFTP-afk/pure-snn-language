#ifndef SNN_STDP_CUH
#define SNN_STDP_CUH

#include "types.h"

// =============================================================================
// STDP（脉冲时序依赖可塑性）kernel 声明
// =============================================================================

// STDP 权重更新（基础版）
// 每个突触独立更新，基于 pre/post 脉冲时序
__global__ void stdp_update_kernel(
    Synapse* synapses,              // [in/out] 突触数组
    int n_synapses,
    const bool* pre_spikes,         // [in] 当前突触前脉冲
    const bool* post_spikes,        // [in] 当前突触后脉冲
    int time_step,
    float dopamine_level            // [in] 多巴胺调制（奖励调制 STDP）
);

// STDP 权重更新（CSR 格式优化版）
__global__ void stdp_update_csr_kernel(
    float* weights,                 // [in/out] 权重
    const int* row_ptr,             // [in] CSR 行指针
    const int* col_idx,             // [in] CSR 列索引
    const int* pre_spike_time,      // [in] 突触前最近脉冲时间 [n_pre]
    const int* post_spike_time,     // [in] 突触后最近脉冲时间 [n_post]
    int n_synapses,
    int time_step,
    float dopamine_level
);

// 权重裁剪（防止爆炸/消失）
__global__ void clip_weights_kernel(
    float* weights,
    int n_synapses,
    float w_min,
    float w_max
);

// Host 端包装函数
void stdp_update(Synapse* d_synapses, int n_synapses,
                 const bool* d_pre_spikes, const bool* d_post_spikes,
                 int time_step, float dopamine_level);
void stdp_update_csr(float* d_weights, const int* d_row_ptr,
                     const int* d_col_idx,
                     const int* d_pre_spike_time,
                     const int* d_post_spike_time,
                     int n_synapses, int time_step, float dopamine);
void clip_weights(float* d_weights, int n_synapses,
                  float w_min, float w_max);

#endif // SNN_STDP_CUH
