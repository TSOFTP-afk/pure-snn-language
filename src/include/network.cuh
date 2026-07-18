#ifndef SNN_NETWORK_CUH
#define SNN_NETWORK_CUH

#include "types.h"

// =============================================================================
// 网络初始化与拓扑结构 kernel 声明
// =============================================================================

// 初始化网络：随机连接 + 局部偏好
// 实现三种拓扑：
// 1. 感觉→联合：稠密投影（输入到中间）
// 2. 联合→联合：局部 + 少量长程
// 3. 联合→运动：稠密投影（中间到输出）
__global__ void init_synapses_kernel(
    Synapse* synapses,
    int* row_ptr,
    int* col_idx,
    int n_neurons,
    int synapses_per_neuron,
    unsigned int seed
);

// Host 端包装函数
void init_synapses(Synapse* d_synapses, int* d_row_ptr, int* d_col_idx,
                   int n_neurons, int synapses_per_neuron, unsigned int seed);

#endif // SNN_NETWORK_CUH
