#ifndef SNN_NEURON_CUH
#define SNN_NEURON_CUH

#include "types.h"

// =============================================================================
// 神经元相关 CUDA kernel 声明
// =============================================================================

// 初始化神经元状态
__global__ void init_neurons_kernel(
    NeuronState* neurons,
    int n_neurons,
    unsigned int seed
);

// LIF 神经元更新（单步）
// 输入：当前神经元状态 + 外部输入电流
// 输出：更新后的状态 + 脉冲输出
__global__ void lif_update_kernel(
    NeuronState* neurons,           // [in/out] 神经元状态
    const float* input_current,     // [in] 外部输入电流
    bool* spike_output,             // [out] 脉冲输出
    int n_neurons,
    int time_step
);

// 更新发放率（滑动平均，用于监控）
__global__ void update_fire_rate_kernel(
    NeuronState* neurons,
    const bool* spike_output,
    int n_neurons,
    float decay
);

// 重置神经元状态（用于 episode 切换）
__global__ void reset_neurons_kernel(
    NeuronState* neurons,
    int n_neurons
);

// Host 端包装函数
void init_neurons(NeuronState* d_neurons, int n_neurons, unsigned int seed);
void lif_update(NeuronState* d_neurons, const float* d_input,
                bool* d_spikes, int n_neurons, int time_step);
void update_fire_rate(NeuronState* d_neurons, const bool* d_spikes,
                      int n_neurons, float decay);
void reset_neurons(NeuronState* d_neurons, int n_neurons);

#endif // SNN_NEURON_CUH
