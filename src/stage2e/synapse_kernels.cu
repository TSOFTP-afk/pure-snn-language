// =============================================================================
// Stage 2e 突触 kernel 实现 (P1)
// =============================================================================
// 设计要点:
//   1. synapse_nmda_kernel:
//      - 每 thread 处理一个突触
//      - 计算突触后神经元的 V_bio, Mg²⁺ 阻塞, NMDA 电导
//      - pre 脉冲时: ampa_conductance 增加 (快速), nmda_conductance 增加 (慢速)
//      - 钙浓度: ca_concentration += nmda_conductance * g_NMDA(V) * dt
//      - 累积 NMDA 电流到 d_nmda_current[post] (atomicAdd)
//      - 每步衰减: nmda_conductance *= exp(-dt/τ_NMDA), ca_concentration *= exp(-dt/τ_ca)
//
//   2. stdp_dual_trace_kernel:
//      - 每 thread 处理一个突触
//      - 更新 x_pre/x_post trace (衰减 + 脉冲跳变)
//      - 计算 Δw = +x_pre * post_spike - x_post * pre_spike
//      - 应用权重更新: weight += η * Δw * plasticity_gain
//      - 兴奋/抑制分别 clamp 到 [0, W_MAX] / [-W_MAX, 0]
//      - 更新 last_pre_spike / last_post_spike (注意: 先算 Δw 再更新 last_spike)
//
//   3. stdp_stp_kernel:
//      - 每 thread 处理一个突触
//      - pre 脉冲时: 更新 utilization (易化) + resource (抑制)
//      - 衰减: resource += (1 - resource) * dt/τ_rec, utilization *= exp(-dt/τ_fac)
//
// 性能考虑:
//   - 10.7M 突触, 每 thread 一个, ~41K blocks × 256 threads
//   - 用 __ldg() 读只读数据
//   - atomicAdd 用于累积 NMDA 电流到 d_nmda_current[post]
// =============================================================================

#include "synapse_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// synapse_nmda_kernel: NMDA 受体电压依赖 + 钙浓度更新
// =============================================================================
__global__ void synapse_nmda_kernel(
    BioSynapse* __restrict__ synapses,
    const NeuronStateAdEx* __restrict__ neurons,
    float* __restrict__ nmda_current,         // 累积到 post 神经元的 NMDA 电流
    float* __restrict__ ca_snapshot,          // 当前步钙快照 (覆盖写入)
    int n_synapses,
    int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    BioSynapse& s = synapses[i];
    int post = s.post_idx;

    // 读取突触后神经元电压
    float V_norm = neurons[post].membrane_potential;
    // 电压重映射: V_bio = -70 + 50 * V_norm
    float V_bio = V_BIO_OFFSET + V_BIO_SCALE * V_norm;

    // Mg²⁺ 阻塞因子 (Jahr & Stevens 1990)
    // g_NMDA(V) = 1 / (1 + [Mg²⁺] * exp(-V_bio/16.13) / 3.57)
    float mg_factor = 1.0f / (1.0f + NMDA_MG_CONCENTRATION *
                               expf(-V_bio / 16.13f) / 3.57f);
    // 当 V_bio < -60mV: mg_factor 接近 0 (NMDA 闭合)
    // 当 V_bio > -20mV: mg_factor 接近 1 (NMDA 开放)

    // 电导衰减 (每步)
    float nmda_decay = expf(-1.0f / NMDA_TAU);     // ~0.993
    float ampa_decay = expf(-1.0f / AMPA_TAU);     // ~0.819
    float ca_decay   = expf(-1.0f / NMDA_CA_TAU);  // ~0.980

    s.nmda_conductance *= nmda_decay;
    s.ampa_conductance *= ampa_decay;
    s.ca_concentration *= ca_decay;

    // 钙浓度更新 (仅 NMDA 开放时)
    float ca_inflow = s.nmda_conductance * mg_factor * 0.01f;
    s.ca_concentration += ca_inflow;
    if (s.ca_concentration > 1.0f) s.ca_concentration = 1.0f;

    // 写入钙快照
    ca_snapshot[i] = s.ca_concentration;

    // 累积 NMDA 电流到 post 神经元 (atomicAdd)
    // I_NMDA = g_NMDA * (V - E_NMDA) * mg_factor
    // E_NMDA ≈ 0 mV, V_norm 归一化, V_bio 重映射
    // 简化: I_NMDA_norm = nmda_cond * mg_factor * (1.4 - V_norm)  (驱动 V_norm 朝 1.4 上升)
    // 仅兴奋性突触贡献 NMDA 电流 (receptor_flags bit1=NMDA)
    if (s.receptor_flags & 0x02) {  // bit1 = NMDA
        float I_nmda = s.nmda_conductance * mg_factor * (1.4f - V_norm) * 0.05f;
        if (fabsf(I_nmda) > 1e-6f) {
            atomicAdd(&nmda_current[post], I_nmda);
        }
    }
}

__global__ void synapse_arrival_conductance_kernel(
    BioSynapse* __restrict__ synapses,
    const int* __restrict__ delay_ring_indices,
    int arrived_ring_idx,
    int arrived_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= arrived_count) return;

    int slot_base = arrived_ring_idx * DELAY_RING_SLOT_CAPACITY;
    int syn_idx = delay_ring_indices[slot_base + i];
    if (syn_idx < 0) return;

    BioSynapse& s = synapses[syn_idx];
    s.ampa_conductance += AMPA_G_MAX * s.resource * s.utilization;
    if (s.receptor_flags & 0x02) {
        s.nmda_conductance += NMDA_G_MAX * s.resource * s.utilization;
    }
}

// =============================================================================
// stdp_dual_trace_kernel: STDP 双 trace + Δw 计算
// =============================================================================
__global__ void stdp_dual_trace_kernel(
    BioSynapse* __restrict__ synapses,
    const bool* __restrict__ spike_flags,
    float* __restrict__ stdp_x_pre_trace,    // 独立数组 (镜像 BioSynapse.x_pre_trace)
    int n_synapses,
    int step,
    float plasticity_gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    BioSynapse& s = synapses[i];
    int post = s.post_idx;

    bool post_spike = spike_flags[post];

    // ----- 1. trace 衰减 -----
    float x_pre_decay  = expf(-1.0f / STDP_X_PRE_TAU);   // ~0.951
    float x_post_decay = expf(-1.0f / STDP_X_POST_TAU);  // ~0.951

    s.x_pre_trace  *= x_pre_decay;
    s.x_post_trace *= x_post_decay;

    // ----- 2. 计算 Δw (必须在更新 last_spike 之前) -----
    // LTP: pre 在 post 之前 → x_pre * post_spike
    // LTD 由延迟到达事件 pass 处理，避免直接读取当前 pre spike
    float delta_w = 0.0f;
    if (post_spike) {
        // post 脉冲: 检查 pre trace (即 x_pre 残留 = pre 是否刚发过)
        delta_w += s.x_pre_trace * STDP_A_PLUS_2E;
    }

    // ----- 3. trace 跳变 (脉冲时) -----
    if (post_spike) {
        s.x_post_trace += STDP_A_MINUS_2E;  // post trace 跳变
        s.last_post_spike = static_cast<float>(step);
    }

    // ----- 4. 应用权重更新 -----
    // P1 阶段: 胚胎期 (0-30K) plasticity_gain=0, 不更新
    // 关键: 先算 Δw 再更新 last_spike (项目记忆硬约束)
    float eta = 0.01f;  // P1 学习率
    s.weight += eta * delta_w * plasticity_gain;

    // ----- 5. 权重 clamp -----
    // 兴奋性: weight > 0, clamp 到 [0, W_MAX]
    // 抑制性: weight < 0, clamp 到 [-W_MAX, 0]
    bool is_exc = (s.receptor_flags & 0x03);  // AMPA|NMDA
    if (is_exc) {
        if (s.weight < 0.0f) s.weight = 0.0f;
        if (s.weight > STDP_W_MAX_2E) s.weight = STDP_W_MAX_2E;
    } else {
        if (s.weight > 0.0f) s.weight = 0.0f;
        if (s.weight < -STDP_W_MAX_2E) s.weight = -STDP_W_MAX_2E;
    }

    // 同步独立 x_pre_trace 数组 (供其他 kernel 读取)
    stdp_x_pre_trace[i] = s.x_pre_trace;
}

__global__ void stdp_arrival_pre_kernel(
    BioSynapse* __restrict__ synapses,
    const int* __restrict__ delay_ring_indices,
    float* __restrict__ stdp_x_pre_trace,
    int arrived_ring_idx,
    int arrived_count,
    int step,
    float plasticity_gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= arrived_count) return;

    int slot_base = arrived_ring_idx * DELAY_RING_SLOT_CAPACITY;
    int syn_idx = delay_ring_indices[slot_base + i];
    if (syn_idx < 0) return;

    BioSynapse& s = synapses[syn_idx];
    float delta_w = -s.x_post_trace * STDP_A_MINUS_2E;
    s.x_pre_trace += STDP_A_PLUS_2E;
    s.last_pre_spike = static_cast<float>(step);

    float eta = 0.01f;
    s.weight += eta * delta_w * plasticity_gain;

    bool is_exc = (s.receptor_flags & 0x03);
    if (is_exc) {
        if (s.weight < 0.0f) s.weight = 0.0f;
        if (s.weight > STDP_W_MAX_2E) s.weight = STDP_W_MAX_2E;
    } else {
        if (s.weight > 0.0f) s.weight = 0.0f;
        if (s.weight < -STDP_W_MAX_2E) s.weight = -STDP_W_MAX_2E;
    }
    stdp_x_pre_trace[syn_idx] = s.x_pre_trace;
}

// =============================================================================
// stdp_stp_kernel: 短期可塑性 (Tsodyks-Markram 1998)
// =============================================================================
__global__ void stdp_stp_kernel(
    BioSynapse* __restrict__ synapses,
    const int* __restrict__ delay_ring_indices,
    int arrived_ring_idx,
    int arrived_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= arrived_count) return;

    int slot_base = arrived_ring_idx * DELAY_RING_SLOT_CAPACITY;
    int syn_idx = delay_ring_indices[slot_base + i];
    if (syn_idx < 0) return;

    BioSynapse& s = synapses[syn_idx];

    // 衰减 (每步)
    float fac_decay = expf(-1.0f / STP_TAU_FAC);  // ~0.995
    float rec_recovery = 1.0f - expf(-1.0f / STP_TAU_REC);  // ~0.002

    // resource 恢复 (朝 1 衰减恢复)
    s.resource += (1.0f - s.resource) * rec_recovery;
    // utilization 衰减 (朝基线 U 衰减)
    float baseline_u = (s.receptor_flags & 0x03) ? STP_U_SE : STP_U_SI;
    s.utilization = baseline_u + (s.utilization - baseline_u) * fac_decay;

    // 到达事件时: STP 更新
    float u_new = s.utilization + baseline_u * (1.0f - s.utilization);
    float r_new = s.resource * (1.0f - u_new);

    s.utilization = u_new;
    s.resource = r_new;
    if (s.utilization > 1.0f) s.utilization = 1.0f;
    if (s.resource < 0.0f) s.resource = 0.0f;
}

// =============================================================================
// Host launchers
// =============================================================================
void launch_synapse_nmda(MemoryAllocator* alloc, int step, int arrived_ring_idx, int arrived_count) {
    PersistentBuffers& b = alloc->buffers();

    // 清零 nmda_current (准备累积)
    cudaMemsetAsync(b.d_nmda_current, 0, N_TOTAL_NEURONS_2E * sizeof(float));

    if (arrived_count > 0) {
        int arrival_blocks = (arrived_count + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        synapse_arrival_conductance_kernel<<<arrival_blocks, THREADS_PER_BLOCK_2E>>>(
            b.d_synapses,
            b.d_delay_ring_indices,
            arrived_ring_idx,
            arrived_count);
    }

    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    synapse_nmda_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses,
        b.d_neurons,
        b.d_nmda_current,
        b.d_ca_snapshot,
        N_TOTAL_SYNAPSES_2E,
        step);
}

void launch_stdp_dual_trace(MemoryAllocator* alloc, int step, float plasticity_gain,
                            int arrived_ring_idx, int arrived_count) {
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    stdp_dual_trace_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses,
        b.d_spike_flags,
        b.d_stdp_x_pre_trace,
        N_TOTAL_SYNAPSES_2E,
        step,
        plasticity_gain);

    if (arrived_count > 0) {
        int arrival_blocks = (arrived_count + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        stdp_arrival_pre_kernel<<<arrival_blocks, THREADS_PER_BLOCK_2E>>>(
            b.d_synapses,
            b.d_delay_ring_indices,
            b.d_stdp_x_pre_trace,
            arrived_ring_idx,
            arrived_count,
            step,
            plasticity_gain);
    }
}

void launch_stdp_stp(MemoryAllocator* alloc, int step, int arrived_ring_idx, int arrived_count) {
    (void)step;
    if (arrived_count <= 0) return;
    PersistentBuffers& b = alloc->buffers();
    int blocks = (arrived_count + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    stdp_stp_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses,
        b.d_delay_ring_indices,
        arrived_ring_idx,
        arrived_count);
}

} // namespace stage2e
