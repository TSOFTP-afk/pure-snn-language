// =============================================================================
// neuron_kernel.cu - LIF 神经元更新 CUDA kernel
// =============================================================================
//
// LIF (Leaky Integrate-and-Fire) 模型：
//   τ_m * dV/dt = -(V - V_rest) + R * I_syn
//   if V >= V_thresh: emit spike, V = V_reset
//
// 离散化（时间步长 dt = TIME_STEP_MS）：
//   V[t+1] = V[t] * beta + I_syn[t]    其中 beta = exp(-dt/τ_m) ≈ 0.95
//
// 局部性：每个神经元独立更新，无数据依赖，完美适配 CUDA
// =============================================================================

#include "neuron.cuh"
#include <curand_kernel.h>

// -----------------------------------------------------------------------------
// 初始化神经元状态
// -----------------------------------------------------------------------------
__global__ void init_neurons_kernel(
    NeuronState* neurons,
    int n_neurons,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;

    // 初始化随机数生成器（用于噪声注入）
    curandState state;
    curand_init(seed, idx, 0, &state);

    NeuronState& n = neurons[idx];
    n.membrane_potential = LIF_REST;
    n.synaptic_current = 0.0f;
    n.last_spike_time = -1000;  // 远古时间，避免误触发 STDP
    n.refractory_remaining = 0;

    // 脑区分配 + 类型划分（每脑区内 80/20 兴奋/抑制）
    // 修复（2026-07-19）：原实现按全局索引划分，导致 Motor 区全为抑制性，
    // 其他脑区全为兴奋性，违反生物学分布
    n.region = get_region(idx);
    n.type = get_neuron_type(idx);

    n.fire_rate = 0.0f;
    n.threshold_offset = 0;   // Homeostatic：初始阈值为基准值
}

// -----------------------------------------------------------------------------
// LIF 神经元更新（核心 kernel）
// -----------------------------------------------------------------------------
__global__ void lif_update_kernel(
    NeuronState* neurons,
    const float* input_current,
    bool* spike_output,
    int n_neurons,
    int time_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;

    NeuronState& n = neurons[idx];

    // 不应期检查：剩余步数 > 0 则跳过更新
    if (n.refractory_remaining > 0) {
        n.refractory_remaining--;
        spike_output[idx] = false;
        // 膜电位衰减
        n.membrane_potential = LIF_REST +
            (n.membrane_potential - LIF_REST) * LIF_BETA;
        return;
    }

    // 1. 突触电流衰减（一阶低通滤波）
    //    I[t+1] = I[t] * beta_I + input[t]
    //    这里简化为与膜电位同样的衰减率
    float beta_i = LIF_BETA;
    n.synaptic_current = n.synaptic_current * beta_i + input_current[idx];

    // 2. 膜电位更新（LIF 核心方程）
    //    V[t+1] = V[t] * beta + I_syn
    //    兴奋性突触：电流为正；抑制性突触：电流为负
    //    这里 input_current 已经包含了正负号
    n.membrane_potential = n.membrane_potential * LIF_BETA + n.synaptic_current;

    // 3. 静息电位拉回（泄漏项）
    if (n.membrane_potential < LIF_REST) {
        n.membrane_potential = LIF_REST;
    }

    // 4. 发放判定（应用 homeostatic 阈值偏移）
    //    实际阈值 = LIF_THRESHOLD + threshold_offset/16.0f
    float effective_threshold = LIF_THRESHOLD + (float)n.threshold_offset / 16.0f;
    bool spike = (n.membrane_potential >= effective_threshold);

    if (spike) {
        // 发放：记录时间，重置膜电位，进入不应期
        n.last_spike_time = time_step;
        n.membrane_potential = LIF_RESET;
        n.refractory_remaining = LIF_REFRACTORY;
        spike_output[idx] = true;
    } else {
        spike_output[idx] = false;
    }
}

// -----------------------------------------------------------------------------
// 更新滑动平均发放率 + Homeostatic plasticity（Intrinsic Plasticity）
//
// IP 规则：threshold_offset 调节使 fire_rate 趋向 target_rate
//   fire_rate > target: 抬高阈值（threshold_offset 增加，更难发放）
//   fire_rate < target: 降低阈值（threshold_offset 减少，更易发放）
//
// 定点存储：threshold_offset 为 signed char，实际值 = stored / 16.0f
//   学习率换算：HOMEOSTATIC_LR（float）→ 每步 stored 增量 = LR * 16
// -----------------------------------------------------------------------------
__global__ void update_fire_rate_kernel(
    NeuronState* neurons,
    const bool* spike_output,
    int n_neurons,
    float decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;

    NeuronState& n = neurons[idx];

    // 滑动平均发放率
    float spike = spike_output[idx] ? 1.0f : 0.0f;
    n.fire_rate = n.fire_rate * decay + spike * (1.0f - decay);

    // Homeostatic: 根据脑区选择目标率
    //   感觉/联合皮层 5Hz（稀疏编码）
    //   运动皮层 20Hz（保证有输出活动）
    float target_rate;
    switch (n.region) {
        case BrainRegion::SENSORY:     target_rate = HOMEOSTATIC_TARGET_RATE_SENSORY;     break;
        case BrainRegion::ASSOCIATION: target_rate = HOMEOSTATIC_TARGET_RATE_ASSOCIATION; break;
        case BrainRegion::MOTOR:       target_rate = HOMEOSTATIC_TARGET_RATE_MOTOR;       break;
        default:                       target_rate = HOMEOSTATIC_TARGET_RATE_SENSORY;     break;
    }

    // delta = LR * (fire_rate - target_rate)  （正值 → 抬高阈值）
    float rate_error = n.fire_rate - target_rate;
    float delta_float = HOMEOSTATIC_LR * rate_error * 16.0f;  // 转换到定点域

    // 累加并 clamp 到 [-2.0, +2.0] → 定点 [-32, +32]
    float new_offset = (float)n.threshold_offset + delta_float;
    float max_stored = HOMEOSTATIC_MAX_OFFSET * 16.0f;  // 32.0
    if (new_offset > max_stored) new_offset = max_stored;
    if (new_offset < -max_stored) new_offset = -max_stored;
    n.threshold_offset = (signed char)new_offset;
}

// -----------------------------------------------------------------------------
// 重置神经元状态（episode 切换时）
// -----------------------------------------------------------------------------
__global__ void reset_neurons_kernel(
    NeuronState* neurons,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;

    NeuronState& n = neurons[idx];
    n.membrane_potential = LIF_REST;
    n.synaptic_current = 0.0f;
    n.last_spike_time = -1000;
    n.refractory_remaining = 0;
    n.fire_rate = 0.0f;
    // 注意：threshold_offset 跨 episode 保留
    // （homeostatic 是慢变量，反映该神经元长期学到的"最佳阈值"）
}

// =============================================================================
// Host 端包装函数
// =============================================================================

void init_neurons(NeuronState* d_neurons, int n_neurons, unsigned int seed) {
    int blocks = (n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_neurons_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_neurons, n_neurons, seed);
    CUDA_CHECK_LAST();
}

void lif_update(NeuronState* d_neurons, const float* d_input,
                bool* d_spikes, int n_neurons, int time_step) {
    int blocks = (n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    lif_update_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_neurons, d_input, d_spikes, n_neurons, time_step);
    CUDA_CHECK_LAST();
}

void update_fire_rate(NeuronState* d_neurons, const bool* d_spikes,
                      int n_neurons, float decay) {
    int blocks = (n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    update_fire_rate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_neurons, d_spikes, n_neurons, decay);
    CUDA_CHECK_LAST();
}

void reset_neurons(NeuronState* d_neurons, int n_neurons) {
    int blocks = (n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    reset_neurons_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_neurons, n_neurons);
    CUDA_CHECK_LAST();
}
