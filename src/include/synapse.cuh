#ifndef SNN_SYNAPSE_CUH
#define SNN_SYNAPSE_CUH

#include "types.h"

// =============================================================================
// 突触相关 kernel 声明
// =============================================================================

// 突触传播：将突触前脉冲传到突触后神经元
// 使用 CSR 格式高效处理稀疏连接
__global__ void synaptic_transmission_kernel(
    const Synapse* synapses,        // [in] 突触数组
    int n_synapses,
    const bool* pre_spikes,         // [in] 突触前脉冲
    float* post_current,            // [out] 突触后电流累加
    int n_post_neurons
);

// 突触传播（CSR 格式优化版）
// CSR: 突触按 post_idx 排序，row_ptr[i] 到 row_ptr[i+1] 是指向神经元 i 的突触
__global__ void synaptic_transmission_csr_kernel(
    const int* row_ptr,             // [in] CSR 行指针 [n_post+1]
    const int* col_idx,             // [in] CSR 列索引（突触前）[n_synapses]
    const float* weights,           // [in] 权重 [n_synapses]
    int n_post,
    const bool* pre_spikes,         // [in] 突触前脉冲 [n_pre]
    float* post_current             // [out] 突触后电流 [n_post]
);

// 初始化突触电流累加器为 0
__global__ void clear_current_kernel(
    float* current,
    int n
);

// 权重同步：d_synapses_.weight → d_weights_
__global__ void sync_weights_kernel(
    float* weights,
    const Synapse* synapses,
    int n_synapses
);

// Host 端包装函数
void synaptic_transmission(const Synapse* d_synapses, int n_synapses,
                           const bool* d_pre_spikes,
                           float* d_post_current, int n_post);
void synaptic_transmission_csr(const int* d_row_ptr, const int* d_col_idx,
                                const float* d_weights, int n_post,
                                const bool* d_pre_spikes,
                                float* d_post_current);
void clear_current(float* d_current, int n);
void sync_weights(float* d_weights, const Synapse* d_synapses, int n_synapses);

#endif // SNN_SYNAPSE_CUH
