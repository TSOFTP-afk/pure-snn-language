// =============================================================================
// Stage 2e 统一调度器实现 (P1)
// =============================================================================
// 对应设计文档 §5.4: v4 流水线
//
// P1 阶段:
//   - 快时间尺度 kernel 全部替换为真实实现 (AdEx + NMDA + STDP + STP)
//   - 中/慢时间尺度仍为占位 (留给 Phase 2-4)
//   - 群体编码输入注入
//   - spike count 统计 (P1 判据: 极差 > 100, 簇状发放出现)
// =============================================================================

#include "scheduler.cuh"
#include "neuron_kernels.cuh"
#include "synapse_kernels.cuh"
#include "input_encoding.cuh"
#include "modulatory_kernels.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>

namespace stage2e {

// P3-C host 端辅助: JS 散度 (bounded [0, ln2])
// JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M=(P+Q)/2
static inline double js_divergence(const double* P, const double* Q, int dim) {
    double js = 0.0;
    for (int b = 0; b < dim; ++b) {
        double pi = P[b], pj = Q[b];
        double m = 0.5 * (pi + pj);
        if (pi > 1e-12 && m > 1e-12) js += 0.5 * pi * log(pi / m);
        if (pj > 1e-12 && m > 1e-12) js += 0.5 * pj * log(pj / m);
    }
    return js < 0.0 ? 0.0 : js;
}

// P1 占位 kernel: 简单清零 (用于未实现的慢时间尺度机制)
__global__ void p1_noop_clear_float(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 0.0f;
}

// P3-D 结构可塑性 (PSW 版本): 证据衰减 + 弱突触重置
// 每 1000 步调用一次, 防止 800K 长测中 α/β 无限增长导致学习僵化
// 1. 证据衰减: α *= decay, β *= decay (保持 α/(α+β) 不变, 但 α+β 减小 → 重获可塑性)
//    生物学意义: "遗忘" = 证据衰减, 让已学习突触重新变得可塑
// 2. 弱突触重置: 当 α+β < evidence_threshold 时, 重置为先验 (α=β=0.5)
//    生物学意义: 证据不足的突触回归无信息先验
// 3. 重新计算 weight = ±W_MAX · α/(α+β) 保持一致性
__global__ void structural_plasticity_decay_kernel(
    BioSynapse* __restrict__ synapses,
    float* __restrict__ synapse_alpha,
    float* __restrict__ synapse_beta,
    int n_synapses,
    float decay_factor,
    float evidence_threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;

    // 1. 证据衰减 (α, β 同步衰减, 保持比例)
    synapse_alpha[i] *= decay_factor;
    synapse_beta[i]  *= decay_factor;

    // 2. 防止退化
    if (synapse_alpha[i] < PSW_ALPHA_MIN) synapse_alpha[i] = PSW_ALPHA_MIN;
    if (synapse_beta[i]  < PSW_BETA_MIN)  synapse_beta[i]  = PSW_BETA_MIN;

    // 3. 弱突触重置 (证据不足 → 回归先验)
    float total_evidence = synapse_alpha[i] + synapse_beta[i];
    if (total_evidence < evidence_threshold) {
        synapse_alpha[i] = PSW_ALPHA_INIT;
        synapse_beta[i]  = PSW_BETA_INIT;
    }

    // 4. 重新计算 weight 保持 PSW 一致性
    BioSynapse& s = synapses[i];
    bool is_exc = (s.receptor_flags & 0x03);
    float w_mag = STDP_W_MAX_2E * synapse_alpha[i] / (synapse_alpha[i] + synapse_beta[i]);
    s.weight = is_exc ? w_mag : -w_mag;
}

// P1 占位 kernel: 统计当前步 spike 数 (用于 P1 判据)
__global__ void count_spikes_kernel(const bool* spike_flags, int n, int* out_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (spike_flags[i]) {
        atomicAdd(out_count, 1);
    }
}

// P2 卡方检验: 累积每个神经元在当前字节下的 spike 计数
// 仅在 input_inject 步调用 (spike 已由 lif_adex 产生)
__global__ void accumulate_neuron_byte_counts_kernel(
    const bool* __restrict__ spike_flags,
    int* __restrict__ neuron_byte_counts,  // [N_TOTAL_NEURONS_2E × 256]
    uint8_t current_byte,
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    if (spike_flags[i]) {
        atomicAdd(&neuron_byte_counts[i * 256 + current_byte], 1);
    }
}

__global__ void p3_inhibitory_competition_kernel(
    const NeuronStateAdEx* __restrict__ neurons,
    float* __restrict__ inhibitory_current,
    int n_neurons,
    float global_drive)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    const NeuronStateAdEx& n = neurons[i];
    float subtype_gain = 1.0f;
    if (n.inhibitory_subtype == 0) subtype_gain = 1.40f;
    else if (n.inhibitory_subtype == 1) subtype_gain = 0.85f;
    else if (n.inhibitory_subtype == 2) subtype_gain = 1.10f;
    // Phase R2 模块 C: region 枚举变更 (旧 3=prefrontal → 新 4=prefrontal, 3=L6)
    // 旧代码 region==3 针对前额叶降低抑制驱动 (0.60), 现迁移至 region==REGION_PREFRONTAL (4)
    float region_gain = (n.region == REGION_PREFRONTAL) ? 0.60f : 1.0f;
    float local_drive = n.fire_rate * 2.0f + global_drive;
    float current = local_drive * subtype_gain * region_gain;
    if (n.neuron_type == 1) current *= 0.35f;
    if (current > 2.0f) current = 2.0f;
    inhibitory_current[i] = current;
}

__global__ void p3_kwta_count_column_spikes_kernel(
    const NeuronStateAdEx* __restrict__ neurons,
    const bool* __restrict__ spike_flags,
    int* __restrict__ column_spikes,
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    if (!spike_flags[i]) return;
    uint8_t col = neurons[i].column_id;
    if (col < N_COLUMNS_2E) {
        atomicAdd(&column_spikes[col], 1);
    }
}

// Phase R2 模块 C (Task 6.1): 仅统计 L6 层 (region==REGION_L6) 神经元的 per-column spike count
// 用于 L6 → 丘脑门控反馈闭环
__global__ void p3_count_l6_column_spikes_kernel(
    const NeuronStateAdEx* __restrict__ neurons,
    const bool* __restrict__ spike_flags,
    int* __restrict__ l6_column_spikes,
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    if (!spike_flags[i]) return;
    if (neurons[i].region != REGION_L6) return;  // 仅 L6
    uint8_t col = neurons[i].column_id;
    if (col < N_COLUMNS_2E) {
        atomicAdd(&l6_column_spikes[col], 1);
    }
}

// Phase R2 模块 C (Task 6.4): 按 region (0=L4, 1=L2/3, 2=L5, 3=L6, 4=prefrontal) 归约
// neuron_byte_counts [N × 256] → layer_byte_responses [5 × 256]
// 用于层间字节选择性统计 (chi-squared 显著神经元比例的输入)
__global__ void p3_layer_byte_response_kernel(
    const int* __restrict__ neuron_byte_counts,  // [N_TOTAL_NEURONS_2E × 256]
    const NeuronStateAdEx* __restrict__ neurons,  // [N_TOTAL_NEURONS_2E]
    int* __restrict__ layer_byte_responses,       // [5 × 256] = 5KB
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    int region = neurons[i].region;
    if (region > 4) return;  // 防御性: 仅 0..4 有效
    const int* row = neuron_byte_counts + (size_t)i * 256;
    int* layer_row = layer_byte_responses + (size_t)region * 256;
    for (int b = 0; b < 256; ++b) {
        int c = row[b];
        if (c > 0) atomicAdd(&layer_row[b], c);
    }
}

// Phase R2 模块 C (Task 6.4): 按 region 分组计算卡方显著神经元统计
// 输出: layer_sig_count[5] (χ² > critical 的神经元数)
//       layer_act_count[5] (row_total >= min_active 的活跃神经元数)
//       layer_chi2_sum[5]  (χ² 累积和, 用于计算均值)
// 注意: double 原子加在 CUDA 上需要 atomicAdd(double*, double) (compute capability >= 6.0)
//       为兼容性, 用 float 累积器代替 (监控指标容忍精度损失)
__global__ void p3_layer_chi2_stats_kernel(
    const int* __restrict__ neuron_byte_counts,    // [N × 256]
    const NeuronStateAdEx* __restrict__ neurons,    // [N]
    const float* __restrict__ injections_per_byte,  // [256] (期望比例, 已归一化)
    int* __restrict__ layer_sig_count,               // [5]
    int* __restrict__ layer_act_count,               // [5]
    float* __restrict__ layer_chi2_sum,              // [5]
    int n_neurons,
    int min_active_total,
    float chi2_critical)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    int region = neurons[i].region;
    if (region > 4) return;

    const int* row = neuron_byte_counts + (size_t)i * 256;
    int row_total = 0;
    for (int b = 0; b < 256; ++b) row_total += row[b];
    if (row_total < min_active_total) return;  // 跳过不活跃神经元

    atomicAdd(&layer_act_count[region], 1);

    // 计算 χ² = Σ_b (observed_b - expected_b)² / expected_b
    // expected_b = row_total * injections_per_byte[b]
    float chi2 = 0.0f;
    for (int b = 0; b < 256; ++b) {
        float expected = (float)row_total * injections_per_byte[b];
        if (expected < 1e-10f) continue;
        float diff = (float)row[b] - expected;
        chi2 += diff * diff / expected;
    }

    atomicAdd(&layer_chi2_sum[region], chi2);
    if (chi2 > chi2_critical) {
        atomicAdd(&layer_sig_count[region], 1);
    }
}

__global__ void p3_kwta_gate_kernel(
    const NeuronStateAdEx* __restrict__ neurons,
    float* __restrict__ inhibitory_current,
    const int* __restrict__ column_spikes,
    int* __restrict__ kwta_stats,
    int n_neurons,
    int target_per_column,
    float inhibition_gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    const NeuronStateAdEx& n = neurons[i];
    uint8_t col = n.column_id;
    if (col >= N_COLUMNS_2E) return;
    int spikes = column_spikes[col];
    if (spikes <= 0) return;
    if (i % NEURONS_PER_COLUMN_2E == 0) {
        atomicAdd(&kwta_stats[0], 1);
        atomicAdd(&kwta_stats[1], spikes < target_per_column ? spikes : target_per_column);
        if (spikes > target_per_column) atomicAdd(&kwta_stats[2], spikes - target_per_column);
    }
    int local_rank_proxy = (i + col * 37) % NEURONS_PER_COLUMN_2E;
    int target_width = target_per_column * 8;
    if (target_width < 1) target_width = 1;
    if (target_width > NEURONS_PER_COLUMN_2E) target_width = NEURONS_PER_COLUMN_2E;
    if (spikes > target_per_column && local_rank_proxy >= target_width) {
        float pressure = (float)(spikes - target_per_column) / (float)(target_per_column + 1);
        float extra = inhibition_gain * pressure;
        if (extra > 2.0f) extra = 2.0f;
        inhibitory_current[i] += extra;
    }
}

__global__ void p3_wm_update_kernel(
    WMSlot* __restrict__ slots,
    float activity_drive,
    int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= WM_SLOTS) return;
    WMSlot& slot = slots[i];
    slot.age += 1;
    slot.activation *= WM_DECAY_FACTOR;
    float phase = ((step / 100 + i * 17) % 100) / 100.0f;
    float candidate = activity_drive * (0.5f + phase);
    if (candidate > WM_ACTIVATION_THRESHOLD && candidate > slot.activation) {
        slot.activation = candidate > 1.0f ? 1.0f : candidate;
        slot.age = 0;
        slot.prefrontal_group = i % PREFRONTAL_GROUPS;
        for (int k = 0; k < WM_PATTERN_DIM; ++k) {
            slot.pattern[k] = slot.activation * ((float)(((i + 1) * (k + 3) + step) % 97) / 96.0f);
        }
    }
}

// P3-C: 每柱字节响应向量归约
// 输入: neuron_byte_counts [N × 256], neurons [N] (取 column_id)
// 输出: column_byte_responses [N_COLUMNS_2E × 256]
// 实现: 每个线程处理一个神经元, 累加其 256 字节计数到所属柱 (atomicAdd)
// 跳过 column_id >= N_COLUMNS_2E 的神经元 (前额叶 column_id=255)
__global__ void p3_column_byte_response_kernel(
    const int* __restrict__ neuron_byte_counts,  // [N_TOTAL_NEURONS_2E × 256]
    const NeuronStateAdEx* __restrict__ neurons, // [N_TOTAL_NEURONS_2E]
    int* __restrict__ column_byte_responses,     // [N_COLUMNS_2E × 256]
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    uint8_t col = neurons[i].column_id;
    if (col >= N_COLUMNS_2E) return;  // 跳过前额叶
    const int* row = neuron_byte_counts + (size_t)i * 256;
    int* col_row = column_byte_responses + (size_t)col * 256;
    for (int b = 0; b < 256; ++b) {
        int c = row[b];
        if (c > 0) atomicAdd(&col_row[b], c);
    }
}

BioMechanismScheduler::BioMechanismScheduler(MemoryAllocator* alloc)
    : alloc_(alloc), stats_{}, delay_ring_idx_(0), last_phase_(-1),
      total_steps_(0), total_spikes_accum_(0), inject_spikes_accum_(0),
      min_spikes_per_step_(INT_MAX), max_spikes_per_step_(0),
      total_burst_steps_(0), total_single_neuron_burst_spikes_(0),
      arrived_events_accum_(0), dispatched_events_accum_(0),
      dropped_events_accum_(0), max_delay_slot_depth_(0),
      p3_inhibitory_updates_(0), p3_wm_updates_(0), p3_last_activity_drive_(0.0f),
      p3_kwta_updates_(0), p3_kwta_active_columns_(0), p3_kwta_winner_estimate_(0),
      p3_kwta_suppressed_estimate_(0), p3_kwta_target_per_column_(5),
      p3_semantic_eval_updates_(0), p3_semantic_eval_last_step_(-1),
      p3_silhouette_score_(0.0), p3_js_divergence_mean_(0.0),
      p3_js_divergence_max_(0.0), p3_column_ratio_(0.0),
      // Phase R2 模块 C (Task 6): L6 反馈 + 层间指标初始化
      d_l6_column_spikes_(nullptr),
      d_layer_byte_responses_(nullptr),
      d_layer_sig_count_(nullptr), d_layer_act_count_(nullptr),
      d_layer_chi2_sum_(nullptr), d_injections_per_byte_(nullptr),
      l6_total_spikes_last_(0), l6_activity_ema_mean_(0.0f),
      d_spike_counter_(nullptr), d_single_neuron_burst_counter_(nullptr),
      d_p3_column_spikes_(nullptr), d_p3_kwta_stats_(nullptr),
      d_p3_column_byte_responses_(nullptr),
      d_gate_states_(nullptr), d_byte_history_(nullptr), d_gate_stats_(nullptr),
      gate_mean_(GATE_INITIAL_SIGNAL), gate_open_ratio_(0.0f) {
    printf("[Stage2e P1] 调度器初始化完成\n");
    memset(&stats_, 0, sizeof(stats_));
    // 分配 device 端 spike 计数器
    CUDA_CHECK_2E(cudaMalloc(&d_spike_counter_, 2 * sizeof(int)));
    d_single_neuron_burst_counter_ = d_spike_counter_ + 1;
    CUDA_CHECK_2E(cudaMemset(d_spike_counter_, 0, 2 * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_p3_column_spikes_, N_COLUMNS_2E * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(d_p3_column_spikes_, 0, N_COLUMNS_2E * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_p3_kwta_stats_, 3 * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(d_p3_kwta_stats_, 0, 3 * sizeof(int)));
    // P3-C: 50 柱 × 256 字节响应缓冲
    CUDA_CHECK_2E(cudaMalloc(&d_p3_column_byte_responses_, N_COLUMNS_2E * 256 * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(d_p3_column_byte_responses_, 0, N_COLUMNS_2E * 256 * sizeof(int)));
    // 丘脑-皮层门控: 分配缓冲区并初始化
    CUDA_CHECK_2E(cudaMalloc(&d_gate_states_, N_COLUMNS_2E * sizeof(ThalamicGateState)));
    CUDA_CHECK_2E(cudaMalloc(&d_byte_history_, 256 * sizeof(unsigned int)));
    CUDA_CHECK_2E(cudaMalloc(&d_gate_stats_, 4 * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(d_byte_history_, 0, 256 * sizeof(unsigned int)));
    CUDA_CHECK_2E(cudaMemset(d_gate_stats_, 0, 4 * sizeof(float)));
    init_thalamic_gate(d_gate_states_);

    // Phase R2 模块 C (Task 6.1): 分配 L6 spike count 缓冲 + 层间指标缓冲
    CUDA_CHECK_2E(cudaMalloc(&d_l6_column_spikes_, N_COLUMNS_2E * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(d_l6_column_spikes_, 0, N_COLUMNS_2E * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_layer_byte_responses_, 5 * 256 * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(d_layer_byte_responses_, 0, 5 * 256 * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_layer_sig_count_, 5 * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(d_layer_sig_count_, 0, 5 * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_layer_act_count_, 5 * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(d_layer_act_count_, 0, 5 * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_layer_chi2_sum_, 5 * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(d_layer_chi2_sum_, 0, 5 * sizeof(float)));
    CUDA_CHECK_2E(cudaMalloc(&d_injections_per_byte_, 256 * sizeof(float)));
    // 初始化为均匀分布 (期望每个字节注入次数相等)
    std::vector<float> h_inj(256, 1.0f / 256.0f);
    CUDA_CHECK_2E(cudaMemcpy(d_injections_per_byte_, h_inj.data(),
                             256 * sizeof(float), cudaMemcpyHostToDevice));
    // host 端层间指标初始化
    for (int i = 0; i < 5; ++i) {
        layer_activation_delay_[i] = 0.0;
        layer_chi2_sig_ratio_[i] = 0.0f;
        layer_chi2_mean_[i] = 0.0;
    }
}

BioMechanismScheduler::~BioMechanismScheduler() {
    printf("[Stage2e P1] 调度器销毁, 共执行 %d 步, 累计脉冲 %d\n",
           total_steps_, total_spikes_accum_);
    if (d_spike_counter_) cudaFree(d_spike_counter_);
    if (d_p3_column_spikes_) cudaFree(d_p3_column_spikes_);
    if (d_p3_kwta_stats_) cudaFree(d_p3_kwta_stats_);
    if (d_p3_column_byte_responses_) cudaFree(d_p3_column_byte_responses_);
    // 丘脑-皮层门控: 释放缓冲区
    if (d_gate_states_) cudaFree(d_gate_states_);
    if (d_byte_history_) cudaFree(d_byte_history_);
    if (d_gate_stats_) cudaFree(d_gate_stats_);
    // Phase R2 模块 C (Task 6): 释放 L6 + 层间指标缓冲
    if (d_l6_column_spikes_) cudaFree(d_l6_column_spikes_);
    if (d_layer_byte_responses_) cudaFree(d_layer_byte_responses_);
    if (d_layer_sig_count_) cudaFree(d_layer_sig_count_);
    if (d_layer_act_count_) cudaFree(d_layer_act_count_);
    if (d_layer_chi2_sum_) cudaFree(d_layer_chi2_sum_);
    if (d_injections_per_byte_) cudaFree(d_injections_per_byte_);
}

void BioMechanismScheduler::step(int current_step) {
    PersistentBuffers& buf = alloc_->buffers();
    const DevPhaseParams& phase = phase_table_.get_params(current_step);

    // ==================== 快时间尺度 (每步) ====================
    // v4 流水线: delay_inject → input_inject → lif_adex → synapse_nmda → stdp_dual_trace → stdp_stp → delay_dispatch

    // 1. 延迟队列注入 (清零 input_current + 注入上一轮 ring_idx 槽位的内容)
    launch_delay_inject(alloc_, delay_ring_idx_);
    int arrived_ring_idx = delay_ring_idx_;
    int arrived_count = delay_queue_last_arrived_events();

    // 1.5 丘脑门控更新 (在 input_inject 之前)
    // 复用 d_p3_column_spikes_ 作为 per-column spike count
    // 此时 d_spike_flags 仍包含上一步的 spike (本步 lif_adex 尚未执行)
    // 清零并重新计算 per-column spike count (上一步的活动)
    {
        int blocks_col = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        cudaMemsetAsync(d_p3_column_spikes_, 0, N_COLUMNS_2E * sizeof(int));
        p3_kwta_count_column_spikes_kernel<<<blocks_col, THREADS_PER_BLOCK_2E>>>(
            buf.d_neurons, buf.d_spike_flags, d_p3_column_spikes_, N_TOTAL_NEURONS_2E);
        // Phase R2 模块 C (Task 6.1): 同时计算 L6 层 per-column spike count
        // 用于 L6 → 丘脑门控反馈闭环 (高 L6 活动 → 门控关闭)
        cudaMemsetAsync(d_l6_column_spikes_, 0, N_COLUMNS_2E * sizeof(int));
        p3_count_l6_column_spikes_kernel<<<blocks_col, THREADS_PER_BLOCK_2E>>>(
            buf.d_neurons, buf.d_spike_flags, d_l6_column_spikes_, N_TOTAL_NEURONS_2E);
    }
    // 计算当前步是否为注入步 + 当前字节 (门控更新需要)
    uint8_t current_byte = 0;
    bool is_inject_step = (current_step % INPUT_INJECT_INTERVAL == 0);
    if (is_inject_step) {
        current_byte = get_byte_for_step(current_step);
    }
    // 门控状态更新 (无论是否注入步都更新, 非注入步用 current_byte=0, novelty=0)
    // Phase R2 模块 C (Task 6.2): 传入 d_l6_column_spikes_ 实现 L6 反馈闭环
    launch_thalamic_gate_update(d_gate_states_, d_p3_column_spikes_,
                                 d_l6_column_spikes_,
                                 current_byte, is_inject_step,
                                 d_byte_history_, d_gate_stats_);

    // 2. 群体编码输入注入 (每 INPUT_INJECT_INTERVAL 步注入一个字节)
    // 门控调制增益: 传入 d_gate_states_ 数组, kernel 内部读取 .gate_signal 字段
    // (ThalamicGateState 是 16B 结构体, gate_signal 在 offset 0, 但相邻柱间隔 16B
    //  故不能用 float* 强转索引, 必须通过结构体字段访问)
    if (is_inject_step) {
        launch_input_inject(alloc_, current_byte, d_gate_states_);
    }

    // 3. AdEx 神经元更新 (产生 spike_flags)
    // Both counters share one allocation; clear them with one launch.
    cudaMemsetAsync(d_spike_counter_, 0, 2 * sizeof(int));
    launch_lif_adex(alloc_, current_step, phase, d_single_neuron_burst_counter_);

    // P2: 字节选择性直方图 (注入步统计 spike count per byte)
    if (is_inject_step) {
        launch_byte_histogram(alloc_, current_byte);
        // P2 卡方检验: 累积每个神经元在当前字节下的 spike 计数
        int blocks_neuron = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        accumulate_neuron_byte_counts_kernel<<<blocks_neuron, THREADS_PER_BLOCK_2E>>>(
            buf.d_spike_flags, buf.d_neuron_byte_counts, current_byte, N_TOTAL_NEURONS_2E);
    }

    // 4. NMDA 受体 + 钙浓度 (pre 侧由延迟到达事件驱动)
    launch_synapse_nmda(alloc_, current_step, arrived_ring_idx, arrived_count);

    // 5. STDP 双 trace + Δw (pre 侧由延迟到达事件驱动)
    launch_stdp_dual_trace(alloc_, current_step, phase.plasticity_gain,
                           arrived_ring_idx, arrived_count);

    // 6. STP 短期可塑性 (由延迟到达事件驱动)
    launch_stdp_stp(alloc_, current_step, arrived_ring_idx, arrived_count);

    // 7. 延迟队列分发 (扫描 spike_flags, 写入 ring[target_ring])
    launch_delay_dispatch(alloc_, current_step, delay_ring_idx_);

    // 8. 抑制性电流清零 (P1: 抑制性网络未实现, 保持 0)
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p1_noop_clear_float<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_inhibitory_current, N_TOTAL_NEURONS_2E);

    // ==================== 中时间尺度 (每 10 步) ====================
    // P2 实现: camkii, eligibility, inhibitory_network
    // E0 消融模式: 跳过 CaMKII 和 eligibility (保留 inhibitory 占位)
    if (current_step % 10 == 0) {
        if (!e0_ablation) {
            launch_camkii_kernel(current_step);
            launch_stdp_eligibility(current_step);
        }
        launch_inhibitory_network(current_step);
    }

    // ==================== 慢时间尺度 (每 100 步) ====================
    // P2 实现: modulatory, scaling, wm_update
    // E0 消融模式: 跳过 modulatory 和 scaling (纯 STDP 不含调质和缩放)
    if (current_step % 100 == 0) {
        if (!e0_ablation) {
            launch_modulatory(current_step);
            launch_scaling(current_step);
        }
        launch_wm_update(current_step);
    }

    // ==================== 极慢时间尺度 (每 1000 步) ====================
    // P1 占位 (Phase 4 实现): structural_plasticity, developmental
    if (current_step % 1000 == 0) {
        launch_structural_plasticity(current_step);
        launch_developmental(current_step);
        // L6 host totals are diagnostic-only and printed every 1000 steps.
        // Avoid forcing a 200-byte device-to-host synchronization every step.
        {
            int h_l6_col[N_COLUMNS_2E]{};
            cudaMemcpy(h_l6_col, d_l6_column_spikes_,
                       sizeof(h_l6_col), cudaMemcpyDeviceToHost);
            l6_total_spikes_last_ = 0;
            for (int c = 0; c < N_COLUMNS_2E; ++c) {
                l6_total_spikes_last_ += h_l6_col[c];
            }
        }
        // 丘脑门控统计: 从 device 拷贝到 host 缓存 (供 main.cpp 读取)
        cudaMemcpy(&gate_mean_, d_gate_stats_, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gate_open_ratio_, d_gate_stats_ + 1, sizeof(float), cudaMemcpyDeviceToHost);
        // Phase R2 模块 C (Task 6.5): 同步 L6 活动 EMA 跨柱均值 (用于 print_step_log)
        // 拷贝 d_gate_states_ (50 柱 × 16B = 800B) 到 host, 计算 l6_activity_ema 的跨柱均值
        {
            std::vector<ThalamicGateState> h_gate(N_COLUMNS_2E);
            cudaMemcpy(h_gate.data(), d_gate_states_,
                       N_COLUMNS_2E * sizeof(ThalamicGateState), cudaMemcpyDeviceToHost);
            float sum = 0.0f;
            for (int c = 0; c < N_COLUMNS_2E; ++c) sum += h_gate[c].l6_activity_ema;
            l6_activity_ema_mean_ = N_COLUMNS_2E > 0 ? sum / N_COLUMNS_2E : 0.0f;
        }
    }

    // ==================== 睡眠重放 (每 10000 步) ====================
    if (current_step % 10000 == 0) {
        launch_replay(current_step);
        // P3-C: 同周期触发语义聚类评估 (silhouette + JS + 柱间差异)
        launch_semantic_eval(current_step);
    }

    // v4: 延迟环形队列指针前进
    delay_ring_idx_ = (delay_ring_idx_ + 1) % DELAY_STEPS_MAX;

    // ==================== 统计 ====================
    // 统计当前步 spike 数 (用于 P1 判据)
    count_spikes_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_spike_flags, N_TOTAL_NEURONS_2E, d_spike_counter_);

    int h_step_counts[2] = {0, 0};
    cudaMemcpy(h_step_counts, d_spike_counter_, sizeof(h_step_counts), cudaMemcpyDeviceToHost);
    const int step_spikes = h_step_counts[0];
    const int step_single_burst = h_step_counts[1];

    total_steps_++;
    total_spikes_accum_ += step_spikes;
    total_single_neuron_burst_spikes_ += step_single_burst;
    stats_.total_spikes = step_spikes;
    stats_.vram_used_bytes = alloc_->vram_used();
    stats_.vram_peak_bytes = alloc_->vram_peak();

    const DelayQueueRuntimeStats& dq = delay_queue_stats();
    arrived_events_accum_ = dq.arrived_events;
    dispatched_events_accum_ = dq.dispatched_events;
    dropped_events_accum_ = dq.dropped_events;
    max_delay_slot_depth_ = dq.max_slot_depth;

    // P1 判据: spike count 极差 + 簇状发放
    if (step_spikes < min_spikes_per_step_) min_spikes_per_step_ = step_spikes;
    if (step_spikes > max_spikes_per_step_) max_spikes_per_step_ = step_spikes;
    // 簇状发放判定 (P1 修正):
    //   原: step_spikes > 2 * avg → 误把"输入注入同步"当簇状发放
    //   改: 排除注入步, 只在非注入步 (网络自主活动) 中统计 burst
    //       非注入步 spike 高于其平均的 1.5 倍 = 网络级联自发活动
    //       (2× 太严格, 实测仅 0.2% burst; 1.5× 实测 ~2% burst)
    is_inject_step = (current_step % INPUT_INJECT_INTERVAL == 0);
    if (!is_inject_step) {
        // 用非注入步的滑动平均作基准
        int non_inject_steps = total_steps_ - (total_steps_ + INPUT_INJECT_INTERVAL - 1) / INPUT_INJECT_INTERVAL;
        if (non_inject_steps > 5) {  // 至少 5 个非注入步才计算平均
            int non_inject_avg = (total_spikes_accum_ - inject_spikes_accum_) / non_inject_steps;
            if (step_spikes > (non_inject_avg * 3) / 2 && non_inject_avg > 0) {
                total_burst_steps_++;
            }
        }
    }
    if (is_inject_step) {
        inject_spikes_accum_ += step_spikes;
    }

    // 日志
    if (current_step % LOG_INTERVAL_2E == 0) {
        print_step_log(current_step);
    }
}

// ==================== P2 kernel 实现 (中时间尺度学习规则) ====================

void BioMechanismScheduler::launch_camkii_kernel(int step) {
    launch_camkii(alloc_, step);
}

void BioMechanismScheduler::launch_stdp_eligibility(int step) {
    ::stage2e::launch_stdp_eligibility(alloc_, step);
}

void BioMechanismScheduler::launch_inhibitory_network(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_neurons || !buf.d_inhibitory_current) return;
    float avg_spikes = total_steps_ > 0 ? (float)total_spikes_accum_ / (float)total_steps_ : 0.0f;
    float activity_drive = stats_.total_spikes / (avg_spikes + 1.0f);
    if (activity_drive > 2.0f) activity_drive = 2.0f;
    if (activity_drive < 0.0f) activity_drive = 0.0f;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p3_inhibitory_competition_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_neurons, buf.d_inhibitory_current, N_TOTAL_NEURONS_2E, activity_drive * 0.15f);
    cudaMemsetAsync(d_p3_column_spikes_, 0, N_COLUMNS_2E * sizeof(int));
    cudaMemsetAsync(d_p3_kwta_stats_, 0, 3 * sizeof(int));
    p3_kwta_count_column_spikes_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_neurons, buf.d_spike_flags, d_p3_column_spikes_, N_TOTAL_NEURONS_2E);

    // 自适应 k: 基于当前活动水平动态调整 k-WTA 的 target_per_column
    // 45K 测试发现 avg_spikes≈61/步, 每柱仅 ~1.2 spike, 固定 k=20 永不触发
    // 自适应策略: k = clamp(avg_spikes_per_column * 0.5, 3, target_per_column)
    float avg_spikes_per_col = avg_spikes / (float)N_COLUMNS_2E;
    int adaptive_k = (int)(avg_spikes_per_col * 0.5f);
    if (adaptive_k < 3) adaptive_k = 3;
    if (adaptive_k > p3_kwta_target_per_column_) adaptive_k = p3_kwta_target_per_column_;

    p3_kwta_gate_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_neurons, buf.d_inhibitory_current, d_p3_column_spikes_, d_p3_kwta_stats_,
        N_TOTAL_NEURONS_2E, adaptive_k, 0.35f);
    int h_kwta_stats[3] = {0, 0, 0};
    cudaMemcpy(h_kwta_stats, d_p3_kwta_stats_, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    p3_kwta_active_columns_ = h_kwta_stats[0];
    p3_kwta_winner_estimate_ = h_kwta_stats[1];
    p3_kwta_suppressed_estimate_ = h_kwta_stats[2];
    p3_kwta_updates_++;
    p3_last_activity_drive_ = activity_drive;
    p3_inhibitory_updates_++;
}

void BioMechanismScheduler::launch_modulatory(int step) {
    // P2: 调质系统 + DA 价值函数
    // 简化信号计算 (基于 spike 统计)
    float avg_spikes = total_steps_ > 0 ? (float)total_spikes_accum_ / total_steps_ : 0.0f;
    float current_spikes = (float)stats_.total_spikes;

    // novelty = 当前 spike 偏离 EMA 的程度
    float novelty = avg_spikes > 0 ? fabsf(current_spikes - avg_spikes) / (avg_spikes + 1.0f) : 0.0f;
    if (novelty > 1.0f) novelty = 1.0f;

    // pred_succ = 简化: spike 在合理范围内表示预测成功
    float pred_succ = (current_spikes > 10.0f && current_spikes < 1000.0f) ? 0.5f : 0.1f;

    // reward = α·novelty + β·pred_succ
    float alpha_r = 0.5f, beta_r = 0.3f;
    float reward = alpha_r * novelty + beta_r * pred_succ;

    // DA 价值函数 (返回 V(s) 和 V(s'))
    float v_s = 0.0f, v_sp = 0.0f;
    launch_da_value_function(alloc_, step, reward, &v_s, &v_sp);

    // TD error: δ = R + γ·V(s') - V(s)
    float da_delta = reward + TD_GAMMA * v_sp - v_s;

    // 调质浓度动力学
    float kl_div = 0.0f;  // P2 简化: NE 暂不触发
    ::stage2e::launch_modulatory(alloc_, step, reward, novelty, pred_succ, kl_div, da_delta);
}

void BioMechanismScheduler::launch_scaling(int step) {
    // P2: 局部突触缩放 (§3.3)
    // 修复: target_fr 必须接近实际平均发放率, 否则缩放会系统性放大/缩小所有权重
    // 200K 测试: avg_fr = 60.75 spikes / 55000 neurons = 1.1 Hz
    // 原值 10Hz >> 1.1Hz 导致 scale_pre=3.0(clamped), 每百步放大 2-5%, 加速饱和
    // 新值 1.0Hz 略低于实际, 让低频神经元轻微放大, 高频神经元轻微缩小
    float target_fr = 1.0f;
    launch_synaptic_scaling(alloc_, step, target_fr);
}

void BioMechanismScheduler::launch_wm_update(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_wm_slots) return;
    float avg_spikes = total_steps_ > 0 ? (float)total_spikes_accum_ / (float)total_steps_ : 0.0f;
    float activity_drive = stats_.total_spikes / (avg_spikes + 1.0f);
    if (activity_drive > 2.0f) activity_drive = 2.0f;
    if (activity_drive < 0.0f) activity_drive = 0.0f;
    int blocks = (WM_SLOTS + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p3_wm_update_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(buf.d_wm_slots, activity_drive, step);
    p3_wm_updates_++;
}

// ==================== P3-C 语义聚类评估 ====================
// 流程:
//   1. device kernel: 把 55K×256 neuron_byte_counts 归约为 50×256 column_byte_responses
//   2. host: 拷贝 50×256 (50KB) 到 host
//   3. host: 归一化为概率分布 P[c][b]
//   4. host: 计算所有柱对的 JS 散度 (50*49/2 = 1225 对)
//   5. host: k-means (k=5) 聚类, 用 JS 距离
//   6. host: silhouette score (用 JS 距离)
//   7. host: 柱间差异 = max_col_total / min_col_total
// 设计文档 §7.1 P3 硬检查点: silhouette > 0.15 + KL > 0.3 (JS≈KL/2, 故 JS > 0.15), 柱间差异 > 2x
void BioMechanismScheduler::launch_semantic_eval(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_neuron_byte_counts || !buf.d_neurons || !d_p3_column_byte_responses_) return;

    const int NC = N_COLUMNS_2E;       // 50
    const int NB = 256;                // 字节数
    const int N  = N_TOTAL_NEURONS_2E; // 55000

    // 1. 清零 + 启动归约 kernel
    cudaMemsetAsync(d_p3_column_byte_responses_, 0, NC * NB * sizeof(int));
    int blocks = (N + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p3_column_byte_response_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_neuron_byte_counts, buf.d_neurons, d_p3_column_byte_responses_, N);

    // 2. 拷贝 50×256 到 host
    std::vector<int> h_col_resp((size_t)NC * NB, 0);
    cudaMemcpy(h_col_resp.data(), d_p3_column_byte_responses_,
               NC * NB * sizeof(int), cudaMemcpyDeviceToHost);

    // 3. 归一化为概率分布
    std::vector<double> P((size_t)NC * NB, 0.0);
    std::vector<double> col_total(NC, 0.0);
    double grand_total = 0.0;
    for (int c = 0; c < NC; ++c) {
        double sum = 0.0;
        for (int b = 0; b < NB; ++b) sum += h_col_resp[(size_t)c * NB + b];
        col_total[c] = sum;
        grand_total += sum;
        if (sum < 1e-10) {
            // 空柱用均匀分布 (避免 NaN)
            for (int b = 0; b < NB; ++b) P[(size_t)c * NB + b] = 1.0 / NB;
        } else {
            for (int b = 0; b < NB; ++b) P[(size_t)c * NB + b] = h_col_resp[(size_t)c * NB + b] / sum;
        }
    }

    // 4. 计算所有柱对 JS 散度
    std::vector<double> js_mat((size_t)NC * NC, 0.0);
    double js_sum = 0.0;
    double js_max = 0.0;
    int js_count = 0;
    for (int i = 0; i < NC; ++i) {
        for (int j = i + 1; j < NC; ++j) {
            double js = js_divergence(&P[(size_t)i * NB], &P[(size_t)j * NB], NB);
            js_mat[(size_t)i * NC + j] = js;
            js_mat[(size_t)j * NC + i] = js;
            js_sum += js;
            if (js > js_max) js_max = js;
            js_count++;
        }
    }
    double js_mean = js_count > 0 ? js_sum / js_count : 0.0;

    // 5. k-means 聚类 (k=5, 用 JS 距离)
    const int K = 5;
    std::vector<int> assign(NC);
    for (int i = 0; i < NC; ++i) assign[i] = i % K;  // 初始化: 轮转分配
    std::vector<double> centroid((size_t)K * NB, 0.0);
    std::vector<double> cent_dist(NC);  // 每个柱到其簇心的 JS 距离

    for (int iter = 0; iter < 20; ++iter) {
        // 更新簇心 (算术平均, 然后归一化)
        std::vector<int> cnt(K, 0);
        std::vector<double> sum_p((size_t)K * NB, 0.0);
        for (int i = 0; i < NC; ++i) {
            int k = assign[i];
            cnt[k]++;
            for (int b = 0; b < NB; ++b) sum_p[(size_t)k * NB + b] += P[(size_t)i * NB + b];
        }
        for (int k = 0; k < K; ++k) {
            if (cnt[k] > 0) {
                double s = 0.0;
                for (int b = 0; b < NB; ++b) {
                    centroid[(size_t)k * NB + b] = sum_p[(size_t)k * NB + b] / cnt[k];
                    s += centroid[(size_t)k * NB + b];
                }
                if (s < 1e-12) {
                    for (int b = 0; b < NB; ++b) centroid[(size_t)k * NB + b] = 1.0 / NB;
                } else {
                    for (int b = 0; b < NB; ++b) centroid[(size_t)k * NB + b] /= s;
                }
            } else {
                // 空簇: 重新初始化为均匀分布
                for (int b = 0; b < NB; ++b) centroid[(size_t)k * NB + b] = 1.0 / NB;
            }
        }
        // 重新分配
        bool changed = false;
        for (int i = 0; i < NC; ++i) {
            double best_d = std::numeric_limits<double>::max();
            int best_k = 0;
            for (int k = 0; k < K; ++k) {
                double d = js_divergence(&P[(size_t)i * NB], &centroid[(size_t)k * NB], NB);
                if (d < best_d) { best_d = d; best_k = k; }
            }
            if (best_k != assign[i]) { assign[i] = best_k; changed = true; }
            cent_dist[i] = best_d;
        }
        if (!changed) break;
    }

    // 6. silhouette score (用 JS 距离)
    // s_i = (b_i - a_i) / max(a_i, b_i)
    // a_i = 同簇其他柱的平均 JS 距离
    // b_i = min 其他簇 的平均 JS 距离
    double sil_sum = 0.0;
    int sil_count = 0;
    for (int i = 0; i < NC; ++i) {
        int ci = assign[i];
        double a_i = 0.0; int a_n = 0;
        double b_i = std::numeric_limits<double>::max();
        for (int k = 0; k < K; ++k) {
            if (k == ci) {
                for (int j = 0; j < NC; ++j) {
                    if (j != i && assign[j] == ci) {
                        a_i += js_mat[(size_t)i * NC + j];
                        a_n++;
                    }
                }
                if (a_n > 0) a_i /= a_n;
            } else {
                double d_sum = 0.0; int d_n = 0;
                for (int j = 0; j < NC; ++j) {
                    if (assign[j] == k) {
                        d_sum += js_mat[(size_t)i * NC + j];
                        d_n++;
                    }
                }
                if (d_n > 0) {
                    double d_mean = d_sum / d_n;
                    if (d_mean < b_i) b_i = d_mean;
                }
            }
        }
        if (a_n > 0 && b_i < std::numeric_limits<double>::max()) {
            double denom = (a_i > b_i) ? a_i : b_i;
            double s = denom > 1e-12 ? (b_i - a_i) / denom : 0.0;
            sil_sum += s;
            sil_count++;
        }
    }
    double silhouette = sil_count > 0 ? sil_sum / sil_count : 0.0;

    // 7. 柱间差异 = max_col_total / min_col_total (仅计入非零柱)
    double col_min = std::numeric_limits<double>::max();
    double col_max = 0.0;
    int nz_cols = 0;
    for (int c = 0; c < NC; ++c) {
        if (col_total[c] > 0.0) {
            if (col_total[c] < col_min) col_min = col_total[c];
            if (col_total[c] > col_max) col_max = col_total[c];
            nz_cols++;
        }
    }
    double column_ratio = (nz_cols >= 2 && col_min > 0.0) ? col_max / col_min : 0.0;

    // 写入成员变量
    p3_silhouette_score_ = silhouette;
    p3_js_divergence_mean_ = js_mean;
    p3_js_divergence_max_ = js_max;
    p3_column_ratio_ = column_ratio;
    p3_semantic_eval_updates_++;
    p3_semantic_eval_last_step_ = step;

    printf("[Stage2e P3-C] step=%d  silhouette=%.4f  js_mean=%.4f  js_max=%.4f  "
           "col_ratio=%.2f  nz_cols=%d/%d  grand_total=%.0f\n",
           step, silhouette, js_mean, js_max, column_ratio, nz_cols, NC, grand_total);

    // =====================================================================
    // Phase R2 模块 C (Task 6.3): 层间激活顺序统计
    // 按 region 分组 (L4/L2-3/L5/L6/prefrontal) 统计每层平均 spike 时间
    // (相对最近一次注入步的延迟), 用于验证皮层前馈扫描顺序 L4 → L2/3 → L5 → L6
    // =====================================================================
    {
        // 最近一次注入步 (eval 步之前的最后一次 inject)
        // inject 步: step % INPUT_INJECT_INTERVAL == 0, 即 step = k * INPUT_INJECT_INTERVAL
        // 最近一次 = floor((step - 1) / INPUT_INJECT_INTERVAL) * INPUT_INJECT_INTERVAL
        int last_inject_step = ((step - 1) / INPUT_INJECT_INTERVAL) * INPUT_INJECT_INTERVAL;
        if (last_inject_step < 0) last_inject_step = 0;

        // 拷贝 neurons 到 host (55K × 56B = 3.08MB, eval 频率 1/10000 步可接受)
        std::vector<NeuronStateAdEx> h_neurons(N);
        cudaMemcpy(h_neurons.data(), buf.d_neurons,
                   N * sizeof(NeuronStateAdEx), cudaMemcpyDeviceToHost);

        // 每层累积延迟和计数 (仅统计 last_spike_time >= last_inject_step 的神经元)
        const char* layer_names[] = {"L4      ", "L2/3    ", "L5      ", "L6      ", "prefront"};
        double layer_delay_sum[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        int    layer_delay_cnt[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < N; ++i) {
            int region = h_neurons[i].region;
            if (region > 4) continue;
            int lst = h_neurons[i].last_spike_time;
            if (lst < last_inject_step) continue;  // 未在最近注入窗口内发放
            layer_delay_sum[region] += (double)(lst - last_inject_step);
            layer_delay_cnt[region]++;
        }
        printf("[Stage2e P3-C] 层间激活延迟 (相对注入步 %d):\n", last_inject_step);
        printf("       %-10s %10s %10s\n", "层级", "平均延迟", "活跃N");
        for (int l = 0; l < 5; ++l) {
            double mean_delay = layer_delay_cnt[l] > 0 ? layer_delay_sum[l] / layer_delay_cnt[l] : -1.0;
            layer_activation_delay_[l] = mean_delay;  // 缓存到成员变量
            printf("       %-10s %10.2f %10d\n", layer_names[l], mean_delay, layer_delay_cnt[l]);
        }
    }

    // =====================================================================
    // Phase R2 模块 C (Task 6.4): 层间字节选择性 (chi-squared 显著神经元比例)
    // 按 region 分组统计每层卡方显著神经元比例
    // 期望: L4 (丘脑输入) 应有更高字节选择性 (更高 sig_ratio)
    // =====================================================================
    {
        // 注入分布: 字节均匀循环注入 (每 INPUT_INJECT_INTERVAL 步一个新字节, 256 字节循环)
        // 故每字节注入次数相等, expected_proportion = 1/256 (已在构造函数初始化)
        // 若 step < 256 * INPUT_INJECT_INTERVAL, 实际注入次数可能不均, 但作为监控指标足够
        // (main.cpp 末尾会做精确的 per-byte 注入次数分析)

        // 清零统计缓冲
        cudaMemsetAsync(d_layer_sig_count_, 0, 5 * sizeof(int));
        cudaMemsetAsync(d_layer_act_count_, 0, 5 * sizeof(int));
        cudaMemsetAsync(d_layer_chi2_sum_, 0, 5 * sizeof(float));

        // 启动 chi2 统计 kernel
        // df=255, p=0.05 → chi2_critical ≈ 293.2 (与 main.cpp 一致)
        // min_active_total=10 (活跃阈值, 与 main.cpp 一致, 避免低活跃神经元产生虚高 chi2)
        int blocks_neuron = (N + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        p3_layer_chi2_stats_kernel<<<blocks_neuron, THREADS_PER_BLOCK_2E>>>(
            buf.d_neuron_byte_counts, buf.d_neurons, d_injections_per_byte_,
            d_layer_sig_count_, d_layer_act_count_, d_layer_chi2_sum_,
            N, 10, 293.2f);

        // 拷贝结果到 host
        int h_sig[5] = {0, 0, 0, 0, 0};
        int h_act[5] = {0, 0, 0, 0, 0};
        float h_chi2_sum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        cudaMemcpy(h_sig, d_layer_sig_count_, 5 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_act, d_layer_act_count_, 5 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_chi2_sum, d_layer_chi2_sum_, 5 * sizeof(float), cudaMemcpyDeviceToHost);

        const char* layer_names[] = {"L4      ", "L2/3    ", "L5      ", "L6      ", "prefront"};
        printf("[Stage2e P3-C] 层间字节选择性 (chi2 > 293.2, df=255):\n");
        printf("       %-10s %8s %8s %10s %10s\n", "layer", "sig", "act", "sig_ratio", "chi2_mean");
        for (int l = 0; l < 5; ++l) {
            float ratio = h_act[l] > 0 ? (float)h_sig[l] / (float)h_act[l] : 0.0f;
            double chi2_mean = h_act[l] > 0 ? (double)h_chi2_sum[l] / (double)h_act[l] : 0.0;
            layer_chi2_sig_ratio_[l] = ratio;      // 缓存到成员变量
            layer_chi2_mean_[l] = chi2_mean;        // 缓存到成员变量
            printf("       %-10s %8d %8d %10.2f %10.2f\n",
                   layer_names[l], h_sig[l], h_act[l], ratio * 100.0f, chi2_mean);
        }
    }
}

void BioMechanismScheduler::launch_structural_plasticity(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_synapses || !buf.d_synapse_alpha || !buf.d_synapse_beta) return;

    // P3-D 结构可塑性 (PSW 版本): 证据衰减 + 弱突触重置
    // 从发育阶段参数获取 prune_threshold (PRUNING 阶段 0.05, 转为证据阈值)
    const DevPhaseParams& phase = phase_table_.get_params(step);
    // 证据阈值: prune_threshold 越大, 越多弱突触被重置为先验
    // PRUNING 阶段 prune_threshold=0.05 → evidence_threshold=0.05 (重置证据 < 0.05 的突触)
    // 其他阶段 prune_threshold=0.0 → evidence_threshold=0.0 (不重置)
    float evidence_threshold = phase.prune_threshold;
    // 衰减因子: 每千步衰减 5% (α/β 同步衰减, 保持比例, 重获可塑性)
    // 修复 L5/L6 chi2 停滞: 原 0.999 (0.1%) 衰减太弱, 前馈权重在 10K 步内饱和
    //   改为 0.95 (5%): 10K 步衰减 40%, 100K 步衰减 99.4%, 定期重置饱和
    float decay_factor = 0.95f;

    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    structural_plasticity_decay_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_synapses, buf.d_synapse_alpha, buf.d_synapse_beta,
        N_TOTAL_SYNAPSES_2E, decay_factor, evidence_threshold);
}

void BioMechanismScheduler::launch_developmental(int step) {
    // 仅打印发育阶段切换
    DevPhase cur = phase_table_.get_phase(step);
    if (static_cast<int>(cur) != last_phase_) {
        print_phase_change(step, cur);
        last_phase_ = static_cast<int>(cur);
    }
}

void BioMechanismScheduler::launch_replay(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_hippo_indices || !buf.d_replay_injection) return;
    // P1 占位 (Phase 4 实现睡眠重放)
}

// ==================== 日志 ====================

void BioMechanismScheduler::print_step_log(int step) {
    const DevPhaseParams& pp = phase_table_.get_params(step);
    DevPhase ph = phase_table_.get_phase(step);
    const char* phase_names[] = {"EMBRYONIC", "SYNAPTOGENIC", "CRITICAL", "PRUNING", "MATURE"};

    // Phase R2 模块 C (Task 6.5): 输出 L6 反馈闭环指标 (l6_spikes, l6_ema, gate)
    printf("[Stage2e P1] step=%6d  phase=%-12s  plast_gain=%.2f  nmda_expr=%.2f  "
           "vram=%.1fMB  spikes=%d  ring=%d  arrived=%d  burst%%=%.1f  "
           "l6_spikes=%d  l6_ema=%.2f  gate=%.3f\n",
           step, phase_names[static_cast<int>(ph)],
           pp.plasticity_gain, pp.nmda_expression,
           stats_.vram_used_bytes / (1024.0 * 1024.0),
           stats_.total_spikes, delay_ring_idx_, delay_queue_stats().last_arrived_events,
           total_steps_ > 0 ? (100.0f * total_burst_steps_ / total_steps_) : 0.0f,
           l6_total_spikes_last_, l6_activity_ema_mean_, gate_mean_);
}

void BioMechanismScheduler::print_phase_change(int step, DevPhase new_phase) {
    const char* phase_names[] = {"EMBRYONIC (0-5K)", "SYNAPTOGENIC (5K-200K)",
                                 "CRITICAL (200K-800K)", "PRUNING (800K-1.5M)",
                                 "MATURE (1.5M-3M)"};
    printf("\n[Stage2e P1] ===== 发育阶段切换: %s (step %d) =====\n\n",
           phase_names[static_cast<int>(new_phase)], step);
}

} // namespace stage2e
