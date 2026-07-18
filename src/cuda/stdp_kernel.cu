// =============================================================================
// stdp_kernel.cu - STDP 突触可塑性 CUDA kernel
// =============================================================================
//
// 实现：Eligibility Trace + 奖励调制 STDP
//
// 核心思想：
//   1. 每步计算 STDP delta_w（基于 pre/post 脉冲时序）
//   2. delta_w 累加到 eligibility trace（带衰减 γ=0.95）
//      这样最近活跃过的突触在未来 ~20 步内仍保留"被加强的资格"
//   3. 权重实际更新 = dopamine × eligibility
//      reward 信号可以延迟到达（如本任务每 25 步给一次），仍能正确归因
//
// 数学：
//   e[t+1] = γ * e[t] + Δw_stdp[t]
//   w[t+1] = w[t] + dopamine * e[t+1]
// =============================================================================

#include "stdp.cuh"
#include <cmath>

#define ELIGIBILITY_DECAY  0.95f   // eligibility trace 衰减率
                                   // γ=0.95 → 20 步后剩余 36%，匹配 25 步模式周期

// -----------------------------------------------------------------------------
// STDP + Eligibility Trace + 多巴胺调制
// 一个 thread 一个突触
// -----------------------------------------------------------------------------
__global__ void stdp_update_kernel(
    Synapse* synapses,
    int n_synapses,
    const bool* pre_spikes,
    const bool* post_spikes,
    int time_step,
    float dopamine_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_synapses) return;

    Synapse& s = synapses[idx];

    bool pre_fired  = pre_spikes[s.pre_idx];
    bool post_fired = post_spikes[s.post_idx];

    // === 第 1 步：基于旧 last_spike 计算 STDP delta_w ===
    float delta_w = 0.0f;

    if (pre_fired) {
        // pre 当前发放，post 之前发放过 → post 先于 pre → LTD
        float dt = s.last_post_spike - (float)time_step;  // ≤ 0
        if (dt < 0.0f && s.last_post_spike > -500.0f) {
            delta_w -= STDP_A_MINUS * expf(dt / STDP_TAU_MINUS);
        }
    }

    if (post_fired) {
        // post 当前发放，pre 之前发放过 → pre 先于 post → LTP
        float dt = (float)time_step - s.last_pre_spike;  // ≥ 0
        if (dt > 0.0f && s.last_pre_spike > -500.0f) {
            delta_w += STDP_A_PLUS * expf(-dt / STDP_TAU_PLUS);
        }
    }

    // === 第 2 步：更新 eligibility trace ===
    // e[t+1] = γ * e[t] + Δw_stdp
    s.eligibility = ELIGIBILITY_DECAY * s.eligibility + delta_w;

    // === 第 3 步：用当前多巴胺水平 × eligibility 更新权重 ===
    // 这样 reward 信号延迟到达时（如本任务每 25 步给一次），
    // 最近活跃过的突触仍能受到 reward 的影响
    float weight_delta = dopamine_level * s.eligibility;

    bool pre_is_excitatory = (get_neuron_type(s.pre_idx) == NeuronType::EXCITATORY);
    if (pre_is_excitatory) {
        s.weight += weight_delta;
        s.weight = fmaxf(0.0f, fminf(STDP_W_MAX, s.weight));
    } else {
        s.weight += weight_delta;
        s.weight = fmaxf(-STDP_W_MAX, fminf(0.0f, s.weight));
    }

    // === 第 4 步：最后才更新 last_spike ===
    if (pre_fired)  s.last_pre_spike  = (float)time_step;
    if (post_fired) s.last_post_spike = (float)time_step;
}

// -----------------------------------------------------------------------------
// CSR 优化版 STDP：一个 thread 一个突触
// 与基础版类似，但权重和连接信息分开存储
// -----------------------------------------------------------------------------
__global__ void stdp_update_csr_kernel(
    float* weights,
    const int* row_ptr,
    const int* col_idx,
    const int* pre_spike_time,
    const int* post_spike_time,
    int n_synapses,
    int time_step,
    float dopamine_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_synapses) return;

    int pre_idx = col_idx[idx];
    // row_ptr 用于反查 post_idx，但这里简化：直接用权重更新
    // 实际 CSR 中 post_idx 由 row_ptr 推得，这里通过参数省略

    int t_pre = pre_spike_time[pre_idx];
    // 注意：post_spike_time 需要外部维护，这里假设已传入

    // 简化版：只在 pre 或 post 刚发放时更新
    // 实际实现中需要 last_pre/last_post 时间，这里省略复杂逻辑
    // 留待优化阶段完善

    float delta_w = 0.0f;
    float dt = (float)(time_step - t_pre);

    if (dt > 0.0f && dt < 100.0f) {  // 时间窗内
        delta_w = STDP_A_PLUS * expf(-dt / STDP_TAU_PLUS) * dopamine_level;
    }

    weights[idx] += delta_w;
    weights[idx] = fmaxf(STDP_W_MIN, fminf(STDP_W_MAX, weights[idx]));
}

// -----------------------------------------------------------------------------
// 权重裁剪（单独调用，防止数值漂移）
// -----------------------------------------------------------------------------
__global__ void clip_weights_kernel(
    float* weights,
    int n_synapses,
    float w_min,
    float w_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_synapses) return;
    weights[idx] = fmaxf(w_min, fminf(w_max, weights[idx]));
}

// =============================================================================
// Host 端包装函数
// =============================================================================

void stdp_update(Synapse* d_synapses, int n_synapses,
                 const bool* d_pre_spikes, const bool* d_post_spikes,
                 int time_step, float dopamine_level) {
    int blocks = (n_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    stdp_update_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_synapses, n_synapses, d_pre_spikes, d_post_spikes,
        time_step, dopamine_level);
    CUDA_CHECK_LAST();
}

void stdp_update_csr(float* d_weights, const int* d_row_ptr,
                     const int* d_col_idx,
                     const int* d_pre_spike_time,
                     const int* d_post_spike_time,
                     int n_synapses, int time_step, float dopamine) {
    int blocks = (n_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    stdp_update_csr_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_weights, d_row_ptr, d_col_idx,
        d_pre_spike_time, d_post_spike_time,
        n_synapses, time_step, dopamine);
    CUDA_CHECK_LAST();
}

void clip_weights(float* d_weights, int n_synapses,
                  float w_min, float w_max) {
    int blocks = (n_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    clip_weights_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_weights, n_synapses, w_min, w_max);
    CUDA_CHECK_LAST();
}
