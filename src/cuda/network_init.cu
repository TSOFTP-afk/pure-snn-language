// =============================================================================
// network_init.cu - 网络拓扑初始化
// =============================================================================
//
// 拓扑设计（与总纲一致）：
//
//   感觉皮层 (N_SENSORY)  →  联合皮层 (N_ASSOCIATION)  →  运动皮层 (N_MOTOR)
//         ↓                         ↓ ( recurrent )             ↓
//         投影                       局部+长程                  投影
//
// 突触生成策略：
//   1. 跨脑区投影：稠密随机连接（感觉→联合，联合→运动）
//   2. 脑区内连接：局部偏好 + 少量长程（联合皮层 recurrent）
//   3. 兴奋/抑制：兴奋性神经元 → 正权重；抑制性 → 负权重
//
// CSR 格式构建：
//   - 按 post_idx 排序突触
//   - 构建 row_ptr（每行起始位置）
// =============================================================================

#include "network.cuh"
#include "synapse.cuh"
#include <curand_kernel.h>

// get_neuron_type() 已在 types.h 中定义（__host__ __device__ inline）

// -----------------------------------------------------------------------------
// 网络初始化 kernel：生成突触连接
// 简化版：均匀随机连接，后续阶段再实现局部偏好
// -----------------------------------------------------------------------------
__global__ void init_synapses_kernel(
    Synapse* synapses,
    int* row_ptr,
    int* col_idx,
    int n_neurons,
    int synapses_per_neuron,
    unsigned int seed
) {
    int post_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (post_idx >= n_neurons) return;

    // 每个突触后神经元生成 synapses_per_neuron 个入边
    int syn_start = post_idx * synapses_per_neuron;
    row_ptr[post_idx] = syn_start;

    curandState state;
    curand_init(seed, post_idx, 0, &state);

    // 决定允许的 pre 脑区（拓扑约束）
    BrainRegion post_region;
    if (post_idx < N_SENSORY_NEURONS) {
        post_region = BrainRegion::SENSORY;
    } else if (post_idx < N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS) {
        post_region = BrainRegion::ASSOCIATION;
    } else {
        post_region = BrainRegion::MOTOR;
    }

    for (int i = 0; i < synapses_per_neuron; i++) {
        int syn_idx = syn_start + i;
        int pre_idx;

        // 根据 post 脑区决定 pre 脑区
        // 简化拓扑：感觉←感觉，联合←感觉+联合，运动←联合
        float r = curand_uniform(&state);

        if (post_region == BrainRegion::SENSORY) {
            // 感觉皮层：自连接（用作记忆回路）
            pre_idx = (int)(curand_uniform(&state) * N_SENSORY_NEURONS);

        } else if (post_region == BrainRegion::ASSOCIATION) {
            // 联合皮层：50% 来自感觉，50% 来自联合
            if (r < 0.5f) {
                pre_idx = (int)(curand_uniform(&state) * N_SENSORY_NEURONS);
            } else {
                pre_idx = N_SENSORY_NEURONS +
                          (int)(curand_uniform(&state) * N_ASSOCIATION_NEURONS);
            }

        } else { // MOTOR
            // 运动皮层：来自联合
            pre_idx = N_SENSORY_NEURONS +
                      (int)(curand_uniform(&state) * N_ASSOCIATION_NEURONS);
        }

        // 确保不连接到自己
        if (pre_idx == post_idx) {
            pre_idx = (pre_idx + 1) % n_neurons;
        }

        synapses[syn_idx].pre_idx = pre_idx;
        synapses[syn_idx].post_idx = post_idx;
        col_idx[syn_idx] = pre_idx;

        // 初始权重：小随机值
        // 兴奋性 pre → 正权重；抑制性 pre → 负权重
        // 修复（2026-07-19）：用 get_neuron_type() 替代全局索引判断，
        // 与 neuron_kernel.cu 的脑区内 80/20 划分保持一致
        bool pre_is_excitatory = (get_neuron_type(pre_idx) == NeuronType::EXCITATORY);
        float base_weight = curand_uniform(&state) * 0.3f;  // [0, 0.3]
        synapses[syn_idx].weight = pre_is_excitatory ? base_weight : -base_weight;

        synapses[syn_idx].delay = 1.0f;  // 固定延迟 1 步
        synapses[syn_idx].last_pre_spike = -1000.0f;
        synapses[syn_idx].last_post_spike = -1000.0f;
        synapses[syn_idx].eligibility = 0.0f;  // Eligibility trace 初始为 0
    }

    // 最后一个元素：总突触数
    if (post_idx == n_neurons - 1) {
        row_ptr[n_neurons] = n_neurons * synapses_per_neuron;
    }
}

// =============================================================================
// Host 端包装函数
// =============================================================================

void init_synapses(Synapse* d_synapses, int* d_row_ptr, int* d_col_idx,
                   int n_neurons, int synapses_per_neuron, unsigned int seed) {
    int blocks = (n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_synapses_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_synapses, d_row_ptr, d_col_idx,
        n_neurons, synapses_per_neuron, seed);
    CUDA_CHECK_LAST();
}
