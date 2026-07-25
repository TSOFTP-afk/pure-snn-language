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

// E0 消融模式 device 开关 (host 端通过 set_e0_ablation 设置)
__device__ bool g_e0_ablation = false;

__device__ __forceinline__ void materialize_trace_pair(
    BioSynapse& s, int* trace_epochs, int i, int step)
{
    int elapsed = step - trace_epochs[i];
    if (elapsed > 0) {
        const float pre_decay = expf(-static_cast<float>(elapsed) / STDP_X_PRE_TAU);
        const float post_decay = expf(-static_cast<float>(elapsed) / STDP_X_POST_TAU);
        s.x_pre_trace *= pre_decay;
        s.x_post_trace *= post_decay;
        trace_epochs[i] = step;
    }
}

// =============================================================================
// synapse_nmda_kernel: NMDA 受体电压依赖 + 钙浓度更新
// =============================================================================
__global__ void nmda_post_state_kernel(
    const NeuronStateAdEx* __restrict__ neurons,
    float2* __restrict__ post_state,
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    float V_norm = neurons[i].membrane_potential;
    float V_bio = V_BIO_OFFSET + V_BIO_SCALE * V_norm;
    float mg_factor = 1.0f / (1.0f + NMDA_MG_CONCENTRATION *
                               expf(-V_bio / 16.13f) / 3.57f);
    post_state[i] = make_float2(V_norm, mg_factor);
}

__global__ void synapse_nmda_kernel(
    BioSynapse* __restrict__ synapses,
    const float2* __restrict__ post_state,
    float* __restrict__ nmda_current,         // 累积到 post 神经元的 NMDA 电流
    float* __restrict__ ca_snapshot,          // 当前步钙快照 (覆盖写入)
    int n_synapses,
    int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    BioSynapse& s = synapses[i];
    int post = s.post_idx;

    // 电压和 Mg²⁺ 阻塞因子只依赖突触后神经元；按神经元计算一次并供其突触复用。
    float2 cached_post_state = post_state[post];
    float V_norm = cached_post_state.x;
    float mg_factor = cached_post_state.y;
    // 当 V_bio < -60mV: mg_factor 接近 0 (NMDA 闭合)
    // 当 V_bio > -20mV: mg_factor 接近 1 (NMDA 开放)

    // 电导衰减 (每步)
    float nmda_decay = expf(-1.0f / NMDA_TAU);     // ~0.993
    float ampa_decay = expf(-1.0f / AMPA_TAU);     // ~0.819
    float ca_decay   = expf(-1.0f / NMDA_CA_TAU);  // ~0.980

    // 前馈连接树突区室化: 基底树突 Ca²⁺ 快速衰减 (模拟 calbindin 缓冲)
    // 生物学原理: 基底树突富含 calbindin-D28k 缓冲蛋白, Ca²⁺ 快速清除, 不易触发回弹 LTD
    // 修复 L5/L6 chi2 停滞: 前馈 Ca²⁺ 上限 0.12 < CA_REBOUND_THRESHOLD 0.15, 回弹 LTD 永不触发
    // 非前馈连接 (反馈/横向/跨柱) 保持顶端树突动力学, 仍受 Ca²⁺ 回弹 LTD 约束
    bool is_feedforward = (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD);
    float ca_decay_ff = is_feedforward ? expf(-1.0f / NMDA_CA_TAU_FEEDFORWARD) : ca_decay;
    float ca_max_ff   = is_feedforward ? CA_MAX_FEEDFORWARD : 1.0f;

    s.nmda_conductance *= nmda_decay;
    s.ampa_conductance *= ampa_decay;
    s.ca_concentration *= ca_decay_ff;

    // 前馈连接: 每步连续恢复 resource (生物学: STP 恢复是连续过程)
    // 修复: 原只在 arrival 时恢复导致 resource 稳态≈0.005, 信号被削弱~200倍
    // 每步恢复让 resource 稳态≈0.13, 有效信号增强~26倍
    if (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD) {
        float rec_recovery = 1.0f - expf(-1.0f / STP_TAU_REC_FEEDFORWARD);
        s.resource += (1.0f - s.resource) * rec_recovery;
    }

    // 钙浓度更新 (仅 NMDA 开放时)
    // P2 修正: 0.01→1.0→1000.0
    //   实测 ca_mean=0.000054, Ca⁴≈8.5e-19, 远低于 CaMKII 激活阈值
    //   增大系数到 1000.0, 让 ca_mean≈0.054, Ca⁴≈8.5e-6, act_ss≈1.36e-4 (>1e-8 判据)
    //   生物学解释: ca_concentration 实际表示归一化的 [Ca²⁺]_local, 系数吸收单位换算
    // 前馈连接: ca_concentration 受 ca_max_ff=0.12 约束, 低于 CA_REBOUND_THRESHOLD=0.15
    //   → 回弹 LTD 永不触发, 前馈权重不被 Ca²⁺ 超载摧毁, L5/L6 持续发放
    float ca_inflow = s.nmda_conductance * mg_factor * 1000.0f;
    s.ca_concentration += ca_inflow;
    if (s.ca_concentration > ca_max_ff) s.ca_concentration = ca_max_ff;

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
// stdp_dual_trace_kernel: STDP 双 trace + Δw 计算 (PSW 版本)
// =============================================================================
// PSW (Probabilistic Synaptic Weights):
//   - 权重作为 Beta(α,β) 分布的期望: w_eff = W_MAX · α/(α+β)
//   - LTP 事件 (delta_w > 0): α += η_α · delta_w · M_ij · plasticity_factor
//   - LTD 事件 (delta_w < 0): β += η_β · |delta_w| · M_ij · plasticity_factor
//   - α, β > 0 恒成立 → w_eff 物理上 ∈ (0, W_MAX), 不可能饱和
//   - α+β = 证据强度 → 自适应学习率衰减 (元可塑性自然涌现)
// =============================================================================
__global__ void stdp_dual_trace_kernel(
    BioSynapse* __restrict__ synapses,
    const int* __restrict__ synapse_post_indices,
    const bool* __restrict__ spike_flags,
    float* __restrict__ stdp_x_pre_trace,    // 独立数组 (镜像 BioSynapse.x_pre_trace)
    int* __restrict__ trace_epochs,
    float* __restrict__ synapse_alpha,       // PSW: LTP 证据
    float* __restrict__ synapse_beta,        // PSW: LTD 证据
    const float* __restrict__ da_conc,       // P2: 三因素调制
    const float* __restrict__ ach_conc,
    const float* __restrict__ ne_conc,
    const float* __restrict__ ht5_conc,
    int n_synapses,
    int step,
    float plasticity_gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    // CSR post indices are contiguous, unlike BioSynapse::post_idx which is
    // separated by an 80-byte stride. This makes the no-event fast path a
    // compact read-only scan.
    int post = synapse_post_indices[i];

    bool post_spike = spike_flags[post];

    // No post event means no LTP, evidence, weight, or eligibility change.
    // Delay trace decay until the synapse is next observed.
    if (!post_spike) return;

    BioSynapse& s = synapses[i];

    // ----- 1. trace 衰减 -----
    materialize_trace_pair(s, trace_epochs, i, step);

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

    // ----- 4. 应用 PSW 权重更新 -----
    // E0 消融: 纯 STDP (M_ij=1, plasticity_factor=1)
    // P2 完整: Δw_final = η · STDP_delta · M_ij(t) · plasticity_factor
    //   M_ij = σ(da_receptor·DA + ach_receptor·ACh + ne_receptor·NE + ht5_receptor·5HT)
    //   plasticity_factor = 1.0 - 0.5·autophosph (CaMKII 巩固)
    float M_ij, plasticity_factor;
    if (g_e0_ablation) {
        M_ij = 1.0f;
        plasticity_factor = 1.0f;
    } else {
        M_ij = 1.0f / (1.0f + expf(-(s.da_receptor * da_conc[post]
                                     + s.ach_receptor * ach_conc[post]
                                     + get_ne_receptor(s) * ne_conc[post]
                                     + get_ht5_receptor(s) * ht5_conc[post])));
        plasticity_factor = 1.0f - 0.5f * s.camkii_autophosph;
    }

    // PSW: delta_w 拆分为 LTP (累加 α) 和 LTD (累加 β)
    // dual_trace_kernel 中只有 LTP 分量 (delta_w >= 0)
    // 前馈连接使用专用学习率 (减慢饱和, 防止 L5/L6 chi2 停滞)
    bool is_feedforward = (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD);
    float eta_alpha = is_feedforward ? PSW_ETA_ALPHA_FEEDFORWARD : PSW_ETA_ALPHA;
    float eta_beta  = is_feedforward ? PSW_ETA_BETA_FEEDFORWARD  : PSW_ETA_BETA;

    if (delta_w > 0.0f) {
        float evidence = eta_alpha * delta_w * plasticity_gain * M_ij * plasticity_factor;
        synapse_alpha[i] += evidence;
    } else if (delta_w < 0.0f) {
        float evidence = eta_beta * (-delta_w) * plasticity_gain * M_ij * plasticity_factor;
        synapse_beta[i] += evidence;
    }

    // Ca²⁺ 回弹 LTD (生物学防饱和核心机制):
    // 当突触局部 Ca²⁺ 超过阈值时, 额外累积 β (LTD 证据)
    // 模拟高频刺激导致的 Ca²⁺ 超载 → 主动削弱突触 (BCM 理论分子基础)
    // 仅当突触后神经元活跃 (post_spike) 时检查, 避免静息突触误触发
    if (post_spike && s.ca_concentration > CA_REBOUND_THRESHOLD) {
        float ca_excess = s.ca_concentration - CA_REBOUND_THRESHOLD;
        float rebound_evidence = eta_beta * ca_excess * CA_REBOUND_LTD_GAIN
                                 * plasticity_gain * M_ij;
        synapse_beta[i] += rebound_evidence;
    }

    // 防止 α/β 退化 (保持 > 0, 否则 w_eff 无定义)
    if (synapse_alpha[i] < PSW_ALPHA_MIN) synapse_alpha[i] = PSW_ALPHA_MIN;
    if (synapse_beta[i]  < PSW_BETA_MIN)  synapse_beta[i]  = PSW_BETA_MIN;

    // 重新计算权重: w_eff = W_MAX · α/(α+β), 抑制性取负
    bool is_exc = (s.receptor_flags & 0x03);  // AMPA|NMDA
    float w_mag = STDP_W_MAX_2E * synapse_alpha[i] / (synapse_alpha[i] + synapse_beta[i]);
    s.weight = is_exc ? w_mag : -w_mag;

    // P2: 累积 STDP delta 到 eligibility (供 stdp_eligibility_kernel 吸收)
    // E0 模式也累积, 但 eligibility kernel 不运行 (scheduler 跳过)
    s.eligibility += delta_w;

    // 同步独立 x_pre_trace 数组 (供其他 kernel 读取)
    stdp_x_pre_trace[i] = s.x_pre_trace;
}

__global__ void stdp_arrival_pre_kernel(
    BioSynapse* __restrict__ synapses,
    const int* __restrict__ delay_ring_indices,
    float* __restrict__ stdp_x_pre_trace,
    int* __restrict__ trace_epochs,
    float* __restrict__ synapse_alpha,        // PSW: LTP 证据
    float* __restrict__ synapse_beta,         // PSW: LTD 证据
    const float* __restrict__ da_conc,        // P2: 三因素调制
    const float* __restrict__ ach_conc,
    const float* __restrict__ ne_conc,
    const float* __restrict__ ht5_conc,
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
    materialize_trace_pair(s, trace_epochs, syn_idx, step);
    int post = s.post_idx;
    float delta_w = -s.x_post_trace * STDP_A_MINUS_2E;
    s.x_pre_trace += STDP_A_PLUS_2E;
    s.last_pre_spike = static_cast<float>(step);

    // P2: 三因素调制 (E0 消融模式下跳过)
    float M_ij, plasticity_factor;
    if (g_e0_ablation) {
        M_ij = 1.0f;
        plasticity_factor = 1.0f;
    } else {
        M_ij = 1.0f / (1.0f + expf(-(s.da_receptor * da_conc[post]
                                     + s.ach_receptor * ach_conc[post]
                                     + get_ne_receptor(s) * ne_conc[post]
                                     + get_ht5_receptor(s) * ht5_conc[post])));
        plasticity_factor = 1.0f - 0.5f * s.camkii_autophosph;
    }

    // PSW: delta_w 拆分为 LTP (累加 α) 和 LTD (累加 β)
    // arrival_pre_kernel 中只有 LTD 分量 (delta_w <= 0)
    // 前馈连接使用专用学习率 (减慢饱和, 防止 L5/L6 chi2 停滞)
    bool is_feedforward = (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD);
    float eta_alpha = is_feedforward ? PSW_ETA_ALPHA_FEEDFORWARD : PSW_ETA_ALPHA;
    float eta_beta  = is_feedforward ? PSW_ETA_BETA_FEEDFORWARD  : PSW_ETA_BETA;

    if (delta_w > 0.0f) {
        float evidence = eta_alpha * delta_w * plasticity_gain * M_ij * plasticity_factor;
        synapse_alpha[syn_idx] += evidence;
    } else if (delta_w < 0.0f) {
        float evidence = eta_beta * (-delta_w) * plasticity_gain * M_ij * plasticity_factor;
        synapse_beta[syn_idx] += evidence;
    }

    // Ca²⁺ 回弹 LTD 已移除: 该机制仅由 stdp_dual_trace_kernel 在 post_spike 时触发
    // (BCM 理论: 回弹 LTD 是 post 端 Ca²⁺ 超载的反应, 应由 post 端事件驱动)
    // pre 到达仅处理标准 STDP 的 LTD 分量 (delta_w = -x_post * A_minus), 不叠加额外 LTD

    if (synapse_alpha[syn_idx] < PSW_ALPHA_MIN) synapse_alpha[syn_idx] = PSW_ALPHA_MIN;
    if (synapse_beta[syn_idx]  < PSW_BETA_MIN)  synapse_beta[syn_idx]  = PSW_BETA_MIN;

    // 重新计算权重: w_eff = W_MAX · α/(α+β), 抑制性取负
    bool is_exc = (s.receptor_flags & 0x03);
    float w_mag = STDP_W_MAX_2E * synapse_alpha[syn_idx] / (synapse_alpha[syn_idx] + synapse_beta[syn_idx]);
    s.weight = is_exc ? w_mag : -w_mag;

    // P2: 累积 STDP delta 到 eligibility
    s.eligibility += delta_w;

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

    // 根据 receptor_flags 选择 STP 参数 (前馈连接用易化型, 其他用抑郁型)
    // bit4 (RECEPTOR_FLAG_FEEDFORWARD) = 前馈连接标志
    bool is_feedforward = (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD);
    float baseline_u, tau_fac, tau_rec;
    if (is_feedforward) {
        // 前馈连接: 易化型 STP (低 U + 快速恢复, 高频下维持信号)
        baseline_u = STP_U_FEEDFORWARD;
        tau_fac = STP_TAU_FAC_FEEDFORWARD;
        tau_rec = STP_TAU_REC_FEEDFORWARD;
    } else if (s.receptor_flags & 0x03) {
        // 兴奋性 (AMPA|NMDA): 抑郁型 STP
        baseline_u = STP_U_SE;
        tau_fac = STP_TAU_FAC;
        tau_rec = STP_TAU_REC;
    } else {
        // 抑制性: 抑郁型 STP
        baseline_u = STP_U_SI;
        tau_fac = STP_TAU_FAC;
        tau_rec = STP_TAU_REC;
    }

    // 衰减 (每步)
    float fac_decay = expf(-1.0f / tau_fac);
    float rec_recovery = 1.0f - expf(-1.0f / tau_rec);

    // resource 恢复 (朝 1 衰减恢复)
    s.resource += (1.0f - s.resource) * rec_recovery;
    // utilization 衰减 (朝基线 U 衰减)
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

    int neuron_blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    nmda_post_state_kernel<<<neuron_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_neurons,
        b.d_nmda_post_state,
        N_TOTAL_NEURONS_2E);

    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    synapse_nmda_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses,
        b.d_nmda_post_state,
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
        b.d_csr_col_idx,
        b.d_spike_flags,
        b.d_stdp_x_pre_trace,
        b.d_stdp_trace_epoch,
        b.d_synapse_alpha,
        b.d_synapse_beta,
        b.d_da_concentration,
        b.d_ach_concentration,
        b.d_ne_concentration,
        b.d_ht5_concentration,
        N_TOTAL_SYNAPSES_2E,
        step,
        plasticity_gain);

    if (arrived_count > 0) {
        int arrival_blocks = (arrived_count + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        stdp_arrival_pre_kernel<<<arrival_blocks, THREADS_PER_BLOCK_2E>>>(
            b.d_synapses,
            b.d_delay_ring_indices,
            b.d_stdp_x_pre_trace,
            b.d_stdp_trace_epoch,
            b.d_synapse_alpha,
            b.d_synapse_beta,
            b.d_da_concentration,
            b.d_ach_concentration,
            b.d_ne_concentration,
            b.d_ht5_concentration,
            arrived_ring_idx,
            arrived_count,
            step,
            plasticity_gain);
    }
}

__global__ void materialize_stdp_traces_kernel(
    BioSynapse* __restrict__ synapses,
    float* __restrict__ stdp_x_pre_trace,
    int* __restrict__ trace_epochs,
    int n_synapses,
    int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;
    BioSynapse& s = synapses[i];
    materialize_trace_pair(s, trace_epochs, i, step);
    stdp_x_pre_trace[i] = s.x_pre_trace;
}

__global__ void reset_stdp_trace_epochs_kernel(int* epochs, int n, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) epochs[i] = step;
}

void materialize_stdp_traces(MemoryAllocator* alloc, int step) {
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    materialize_stdp_traces_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses, b.d_stdp_x_pre_trace, b.d_stdp_trace_epoch,
        N_TOTAL_SYNAPSES_2E, step);
}

void reset_stdp_trace_epochs(MemoryAllocator* alloc, int step) {
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    reset_stdp_trace_epochs_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_stdp_trace_epoch, N_TOTAL_SYNAPSES_2E, step);
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

// =============================================================================
// P2: CaMKII 自磷酸化动力学 (Graupner & Brunel 2012, 每10步)
// =============================================================================
__global__ void camkii_kernel(
    BioSynapse* __restrict__ synapses,
    float* __restrict__ camkii_activity,
    int n_synapses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    BioSynapse& s = synapses[i];
    float ca = s.ca_concentration;
    float act = camkii_activity[i];
    float autophosph = s.camkii_autophosph;

    // 全局 PP1 水平 (简化为常数 0.5)
    float pp1 = 0.5f;

    // d(activity)/dt = +k1·Ca^4·(1-activity) - k2·activity·PP1
    float ca4 = ca * ca * ca * ca;
    float d_act = CAMKII_K1 * ca4 * (1.0f - act) - CAMKII_K2 * act * pp1;

    // d(autophosph)/dt = +k3·activity^2·(1-autophosph) - k4·autophosph·PP1
    float d_auto = CAMKII_K3 * act * act * (1.0f - autophosph)
                 - CAMKII_K4 * autophosph * pp1;

    // 10步累积 (dt=10)
    act += d_act * 10.0f;
    autophosph += d_auto * 10.0f;

    // clamp
    if (act < 0.0f) act = 0.0f;
    if (act > 1.0f) act = 1.0f;
    if (autophosph < 0.0f) autophosph = 0.0f;
    if (autophosph > 1.0f) autophosph = 1.0f;

    camkii_activity[i] = act;
    s.camkii_autophosph = autophosph;
}

void launch_camkii(MemoryAllocator* alloc, int step) {
    (void)step;
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    camkii_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses, b.d_camkii_activity, N_TOTAL_SYNAPSES_2E);
}

// =============================================================================
// P2: 2阶 eligibility trace (每10步)
// e1(t) = λ1·e1(t-1) + STDP_delta(t)  快 τ~20ms
// e2(t) = λ2·e2(t-1) + e1(t)          慢 τ~200ms
// =============================================================================
__global__ void stdp_eligibility_kernel(
    BioSynapse* __restrict__ synapses,
    float* __restrict__ eligibility,       // e1 (独立数组)
    float* __restrict__ eligibility_slow,  // e2 (独立数组)
    int n_synapses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    // 衰减率 (10步累积)
    float e1_decay = expf(-10.0f / STDP_E1_TAU);
    float e2_decay = expf(-10.0f / STDP_E2_TAU);

    // e1 衰减 + 从 BioSynapse.eligibility 吸收 STDP delta
    float e1 = eligibility[i] * e1_decay + synapses[i].eligibility;
    // e2 衰减 + 吸收 e1
    float e2 = eligibility_slow[i] * e2_decay + e1;

    // 清零 BioSynapse.eligibility (已被吸收)
    synapses[i].eligibility = 0.0f;

    eligibility[i] = e1;
    eligibility_slow[i] = e2;
}

void launch_stdp_eligibility(MemoryAllocator* alloc, int step) {
    (void)step;
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    stdp_eligibility_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses, b.d_eligibility, b.d_eligibility_slow, N_TOTAL_SYNAPSES_2E);
}

__global__ void camkii_eligibility_kernel(
    BioSynapse* __restrict__ synapses,
    float* __restrict__ camkii_activity,
    float* __restrict__ eligibility,
    float* __restrict__ eligibility_slow,
    int n_synapses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    BioSynapse& s = synapses[i];

    // CaMKII update (identical to camkii_kernel).
    float ca = s.ca_concentration;
    float act = camkii_activity[i];
    float autophosph = s.camkii_autophosph;
    float ca4 = ca * ca * ca * ca;
    float d_act = CAMKII_K1 * ca4 * (1.0f - act) - CAMKII_K2 * act * 0.5f;
    float d_auto = CAMKII_K3 * act * act * (1.0f - autophosph)
                 - CAMKII_K4 * autophosph * 0.5f;
    act += d_act * 10.0f;
    autophosph += d_auto * 10.0f;
    if (act < 0.0f) act = 0.0f;
    if (act > 1.0f) act = 1.0f;
    if (autophosph < 0.0f) autophosph = 0.0f;
    if (autophosph > 1.0f) autophosph = 1.0f;
    camkii_activity[i] = act;
    s.camkii_autophosph = autophosph;

    // Eligibility update (identical to stdp_eligibility_kernel).
    float e1_decay = expf(-10.0f / STDP_E1_TAU);
    float e2_decay = expf(-10.0f / STDP_E2_TAU);
    float e1 = eligibility[i] * e1_decay + s.eligibility;
    float e2 = eligibility_slow[i] * e2_decay + e1;
    s.eligibility = 0.0f;
    eligibility[i] = e1;
    eligibility_slow[i] = e2;
}

void launch_camkii_eligibility(MemoryAllocator* alloc, int step) {
    (void)step;
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    camkii_eligibility_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_synapses,
        b.d_camkii_activity,
        b.d_eligibility,
        b.d_eligibility_slow,
        N_TOTAL_SYNAPSES_2E);
}

// =============================================================================
// P2: 局部突触缩放 (§3.3, 每100步)
// scale_local(i) = (target_fr / mean_FR(i))^α
// w_ij *= scale_i · clamp(scale_j / scale_i, 0.5, 2.0)
// =============================================================================
__global__ void synaptic_scaling_kernel(
    BioSynapse* __restrict__ synapses,
    const NeuronStateAdEx* __restrict__ neurons,
    int n_synapses,
    float target_fr,
    float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    BioSynapse& s = synapses[i];
    int pre = s.pre_idx;
    int post = s.post_idx;

    // 读取 pre/post 的发放率
    float fr_pre  = neurons[pre].fire_rate;
    float fr_post = neurons[post].fire_rate;

    // 避免除零
    if (fr_pre  < 0.001f) fr_pre  = 0.001f;
    if (fr_post < 0.001f) fr_post = 0.001f;

    // 局部缩放因子
    float scale_pre  = powf(target_fr / fr_pre,  alpha);
    float scale_post = powf(target_fr / fr_post, alpha);

    // clamp scale_pre/scale_post 到合理范围
    if (scale_pre  > 3.0f) scale_pre  = 3.0f;
    if (scale_pre  < 0.3f) scale_pre  = 0.3f;
    if (scale_post > 3.0f) scale_post = 3.0f;
    if (scale_post < 0.3f) scale_post = 0.3f;

    // 耦合约束: clamp(scale_post / scale_pre, 0.5, 2.0)
    float ratio = scale_post / scale_pre;
    if (ratio < 0.5f) ratio = 0.5f;
    if (ratio > 2.0f) ratio = 2.0f;

    // 最终缩放
    float final_scale = scale_pre * ratio * 0.01f;  // 0.01 = 缩放步长 (防止突变)

    // 缓慢应用 (每100步只微调)
    s.weight *= (1.0f - 0.01f + final_scale);

    // 保存缩放因子
    s.scaling_factor = final_scale;

    // clamp 权重
    bool is_exc = (s.receptor_flags & 0x03);
    if (is_exc) {
        if (s.weight < 0.0f) s.weight = 0.0f;
        if (s.weight > STDP_W_MAX_2E) s.weight = STDP_W_MAX_2E;
    } else {
        if (s.weight > 0.0f) s.weight = 0.0f;
        if (s.weight < -STDP_W_MAX_2E) s.weight = -STDP_W_MAX_2E;
    }
}

void launch_synaptic_scaling(MemoryAllocator* alloc, int step, float target_fr) {
    (void)step;
    (void)target_fr;
    // PSW 模式下跳过 synaptic_scaling
    // 原因: PSW 的 α/β 自适应学习率衰减已提供稳态机制
    //       (α+β 大 → 学习率自动降低, 高活动突触自然稳定)
    //       synaptic_scaling 直接修改 s.weight 会破坏 α/β 与 weight 的一致性
    //       (w_eff 必须始终等于 W_MAX · α/(α+β))
    // 保留 synaptic_scaling_kernel 代码供未来消融对比实验用
    (void)alloc;
}

// E0 消融模式: 设置 device 开关
void set_e0_ablation(bool enable) {
    cudaError_t err = cudaMemcpyToSymbol(g_e0_ablation, &enable, sizeof(bool));
    if (err != cudaSuccess) {
        printf("[E0] cudaMemcpyToSymbol 失败: %s\n", cudaGetErrorString(err));
    } else {
        printf("[E0] device 开关已设置: g_e0_ablation = %s\n", enable ? "true" : "false");
    }
}

} // namespace stage2e
