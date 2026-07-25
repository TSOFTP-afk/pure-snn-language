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
#include "pca_kernels.cuh"
#include "hippocampal_kernels.cuh"
#include "coactivation_kernels.cuh"
#include "wm_kernels.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
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

// Task 2: spike_flags (bool) → 发放率 (float) 转换, 供 PCA 签名提取用
// 联合皮层前 N_ASSOCIATION_NEURONS_2E 个神经元的瞬时发放 (0/1) 转为 float
__global__ void pca_spike_to_float_kernel(const bool* __restrict__ spikes,
                                          float* __restrict__ fr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr[i] = spikes[i] ? 1.0f : 0.0f;
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

    // Task 2: PCA 集成 — CPU 端镜像 + GPU 辅助缓冲初始化
    // h_pca_W_ [50K × 50 = 2.5M float = 10MB]
    // P0 修复: 对称性打破 — Oja's rule 要求 W 非零初值, 否则 proj=Σ W·x=0 导致
    //         更新量 η·(x - W·proj)·proj = η·x·0 = 0, W 永久死锁在 0
    //         修复: 初始化为小随机值 N(0, σ²), σ=0.01 (远小于 STDP_W_MAX=1.5)
    h_pca_W_.resize((size_t)N_ASSOCIATION_NEURONS_2E * PCA_N_COMPONENTS, 0.0f);
    {
        // 固定种子保证可复现 (不依赖 topology_seed, PCA 初始化独立)
        std::mt19937 pca_rng(42);
        std::normal_distribution<float> pca_dist(0.0f, 0.01f);
        for (auto& w : h_pca_W_) w = pca_dist(pca_rng);
    }
    h_fr_snapshot_.resize(N_ASSOCIATION_NEURONS_2E, 0.0f);
    h_mean_fr_.resize(N_ASSOCIATION_NEURONS_2E, 0.0f);
    h_spike_buf_.resize(N_ASSOCIATION_NEURONS_2E, 0);
    // GPU 辅助缓冲: 供 compute_pca_signature / pca_back_project 调用 PCA kernel
    CUDA_CHECK_2E(cudaMalloc(&d_pca_fr_, N_ASSOCIATION_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(d_pca_fr_, 0, N_ASSOCIATION_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMalloc(&d_pca_mean_, N_ASSOCIATION_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(d_pca_mean_, 0, N_ASSOCIATION_NEURONS_2E * sizeof(float)));
    // P0 修复: 同步打破对称性后的 h_pca_W_ 到 GPU d_pca_W (确保首次 compute_pca_signature 有效)
    {
        PersistentBuffers& buf = alloc_->buffers();
        if (buf.d_pca_W) {
            CUDA_CHECK_2E(cudaMemcpy(buf.d_pca_W, h_pca_W_.data(),
                                     (size_t)N_ASSOCIATION_NEURONS_2E * PCA_N_COMPONENTS * sizeof(float),
                                     cudaMemcpyHostToDevice));
        }
    }

    // Task 4-5: 睡眠重放临时缓冲
    CUDA_CHECK_2E(cudaMalloc(&d_replay_sig_, PCA_N_COMPONENTS * sizeof(float)));
    CUDA_CHECK_2E(cudaMalloc(&d_replay_recon_, N_ASSOCIATION_NEURONS_2E * sizeof(float)));
    // Task 8: 结构可塑性临时缓冲
    CUDA_CHECK_2E(cudaMalloc(&d_new_synapse_pairs_, 2 * COACT_MAX_NEW_SYNAPSES * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_new_synapse_count_, sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_new_modulator_scores_, COACT_MAX_NEW_SYNAPSES * sizeof(float)));
    CUDA_CHECK_2E(cudaMalloc(&d_prune_marks_, (size_t)N_TOTAL_SYNAPSES_2E * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_prune_count_, sizeof(int)));
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
    // Task 2: 释放 PCA GPU 辅助缓冲
    if (d_pca_fr_) cudaFree(d_pca_fr_);
    if (d_pca_mean_) cudaFree(d_pca_mean_);
    // Task 4-5: 释放睡眠重放临时缓冲
    if (d_replay_sig_) cudaFree(d_replay_sig_);
    if (d_replay_recon_) cudaFree(d_replay_recon_);
    // Task 8: 释放结构可塑性临时缓冲
    if (d_new_synapse_pairs_) cudaFree(d_new_synapse_pairs_);
    if (d_new_synapse_count_) cudaFree(d_new_synapse_count_);
    if (d_new_modulator_scores_) cudaFree(d_new_modulator_scores_);
    if (d_prune_marks_) cudaFree(d_prune_marks_);
    if (d_prune_count_) cudaFree(d_prune_count_);
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
    // Task 18: 睡眠重放期间 (is_sleeping_=true) 跳过外部字节注入, 防止污染重放模式
    if (is_inject_step && !is_sleeping_) {
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

    // 9. Task 7: L5 → 运动皮层突触传递 + 运动皮层 AdEx 更新
    //    - 主网络 spike_flags 已由 step 3 (lif_adex) 写入, 可作为 L5 突触前信号源
    //    - L5→Motor 突触传递: 把 L5 脉冲通过 CSR 突触转为运动皮层输入电流
    //    - 运动皮层 AdEx 更新: 复用 lif_adex_kernel, 用独立缓冲 (d_motor_neurons/spike_flags)
    //    - 运动皮层脉冲不写入主 d_spike_flags, 不干扰主网络 STDP/NMDA/统计
    //    时序: 在主网络完整更新 (含 delay_dispatch) 之后, 中时间尺度之前
    launch_l5_to_motor_synapse(alloc_);
    launch_motor_adex(alloc_, current_step, phase);

    // ==================== 中时间尺度 (每 10 步) ====================
    // P2 实现: camkii, eligibility, inhibitory_network
    // E0 消融模式: 跳过 CaMKII 和 eligibility (保留 inhibitory 占位)
    if (current_step % 10 == 0) {
        if (!e0_ablation) {
            launch_camkii_eligibility(alloc_, current_step);
        }
        launch_inhibitory_network(current_step);
    }

    // ==================== Task 8: 共激活采样 (每步) ====================
    // 在突触传递完成后, 从当前发放神经元采样候选对, 更新 d_coact_trackers
    if (!e0_ablation && buf.d_coact_trackers) {
        // P0 修复: stats_.da_level 已在每 100 步 modulatory 更新后同步 (line 607)
        //          此处直接读取, 避免每步调用 get_modulatory_stats 的开销
        float current_da = stats_.da_level;
        launch_coactivation_sample(
            buf.d_coact_trackers, buf.d_tracker_count,
            buf.d_spike_flags, current_da,
            N_TOTAL_NEURONS_2E, COACT_TRACKER_SIZE, COACT_SAMPLE_SIZE,
            42u,  // 固定种子 (xorshift32 内部会混入 tid 和 step)
            current_step);
        coact_sample_count_++;
    }

    // P2 修复: 共激活计数衰减 (每 COACT_DECAY_INTERVAL 步)
    // 对所有 tracker 执行 coact_count *= COACT_DECAY_FACTOR, 低频对自然归零被淘汰
    if (!e0_ablation && buf.d_coact_trackers &&
        current_step % COACT_DECAY_INTERVAL == 0 && current_step > 0) {
        launch_coactivation_decay(buf.d_coact_trackers, buf.d_tracker_count,
                                  COACT_TRACKER_SIZE);
        // 衰减后立即清理已归零的陈旧条目
        launch_coactivation_prune(buf.d_coact_trackers, buf.d_tracker_count,
                                  COACT_TRACKER_SIZE, current_step,
                                  COACT_STALE_THRESHOLD);
    }

    // ==================== 慢时间尺度 (每 100 步) ====================
    // P2 实现: modulatory, scaling, wm_update
    // E0 消融模式: 跳过 modulatory 和 scaling (纯 STDP 不含调质和缩放)
    // Task 18: 睡眠态跳过 modulatory (ACh/DA 刷新), 让 ACh 保持低水平巩固模式
    if (current_step % 100 == 0) {
        if (!e0_ablation && !is_sleeping_) {
            launch_modulatory(current_step);
            launch_scaling(current_step);
            // P0 修复: 同步 modulatory 系统的 DA/ACh 浓度到 stats_ (供共激活采样和 Task 18 使用)
            // NetworkStats2e.da_level/ach_level 原本从未被赋值, 导致 modulator_score=0
            ModulatoryStats mod_stats = get_modulatory_stats(alloc_);
            stats_.da_level  = mod_stats.da_mean;
            stats_.ach_level = mod_stats.ach_mean;
            stats_.ne_level  = mod_stats.ne_mean;
            stats_.ht5_level = mod_stats.ht5_mean;
        }
        launch_wm_update(current_step);
    }

    // ==================== Task 2: PCA 增量更新 (每 PCA_UPDATE_INTERVAL 步) ====================
    // warmup 期 (step <= PCA_WARMUP_STEPS) 不更新, 等发放率稳定
    // CPU 端 Oja's rule 在线学习 h_pca_W_, 每 PCA_SYNC_INTERVAL 步同步到 GPU
    if (current_step > PCA_WARMUP_STEPS && current_step % PCA_UPDATE_INTERVAL == 0) {
        launch_pca_update_cpu(current_step);
    }

    // ==================== Task 3: 海马索引编码 (每 HIPP_ENCODE_INTERVAL 步) ====================
    if (current_step > PCA_WARMUP_STEPS && current_step % HIPP_ENCODE_INTERVAL == 0) {
        // 计算 PCA 签名 (复用 compute_pca_signature)
        compute_pca_signature(d_replay_sig_);
        // 海马编码: 新颖模式 LRU 写入 / 已有模式 importance 刷新
        launch_hippo_encode(
            buf.d_hippo_indices, d_replay_sig_,
            buf.d_hippo_write_cursor, buf.d_hippo_filled_count,
            current_step, HIPP_INDEX_SIZE, HIPP_NOVELTY_THRESHOLD);
        // P1.2 修复: 时间衰减 — 编码后对所有已填充索引执行 importance *= HIPP_TIME_DECAY
        // 需读取当前 filled_count (host 端), 然后 GPU 端 grid 跨步衰减
        {
            int h_filled = 0;
            CUDA_CHECK_2E(cudaMemcpy(&h_filled, buf.d_hippo_filled_count,
                                     sizeof(int), cudaMemcpyDeviceToHost));
            launch_hippo_time_decay(buf.d_hippo_indices, h_filled, HIPP_INDEX_SIZE);
        }
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

    // ==================== Task 4-5: 在线解码 (前向预测 + 误差驱动学习) ====================
    // 在所有现有 kernel 之后执行 (spike_flags 已由 lif_adex 写入, 延迟队列已分发)
    // decode_step 内部:
    //   1. 每步: forward + softmax + argmax (预测)
    //   2. 仅注入步 + K 步 warmup 后: error + weight_update (学习)
    //   3. 每 100 步: weight_normalize (正则化)
    // Task 10: decode_update_weights 由 main.cpp 根据 config.eval_mode 设置
    //   eval_mode=true → decode_update_weights=false → 仅前向预测, 不更新 W_decode
    decode_step(current_byte, is_inject_step, /*update_weights=*/decode_update_weights);

    // ==================== 统计 ====================
    // 统计当前步 spike 数 (用于 P1 判据)
    count_spikes_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_spike_flags, N_TOTAL_NEURONS_2E, d_spike_counter_);

    finish_delay_dispatch();

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

    // Task 6: 从 d_decode_error 计算在线解码预测误差 L2 范数
    // ||error|| = sqrt(Σ_b error[b]²), ∈ [0, sqrt(2)] ≈ [0, 1.414]
    //   - 完美预测 (softmax 完全匹配 one-hot): ||error|| = 0 → DA 最高
    //   - 均匀预测 (softmax = 1/256):          ||error|| ≈ 1.0 → DA 中性
    //   - 完全错误 (softmax 给错类概率 1):      ||error|| = sqrt(2) → DA 最低
    // warmup 期间 (input_byte_history_count_ < PREDICTION_DELAY_STEPS) 未产生真实误差,
    //   d_decode_error 全零 (||error||=0) 会误判为完美预测, 用中性值 1.0 代替
    float prediction_error_norm = 1.0f;  // 默认中性 (warmup 或未解码)
    PersistentBuffers& buf = alloc_->buffers();
    if (buf.d_decode_error && input_byte_history_count_ >= PREDICTION_DELAY_STEPS) {
        float h_error[256];
        CUDA_CHECK_2E(cudaMemcpy(h_error, buf.d_decode_error,
                                  256 * sizeof(float), cudaMemcpyDeviceToHost));
        float sq_sum = 0.0f;
        for (int b = 0; b < 256; ++b) {
            sq_sum += h_error[b] * h_error[b];
        }
        prediction_error_norm = sqrtf(sq_sum);
        // clamp 到 [0, sqrt(2)] 防止数值异常
        if (prediction_error_norm > 1.41421356f) prediction_error_norm = 1.41421356f;
    }

    // 调质浓度动力学
    float kl_div = 0.0f;  // P2 简化: NE 暂不触发
    ::stage2e::launch_modulatory(alloc_, step, reward, novelty, pred_succ, kl_div, da_delta,
                                  prediction_error_norm);
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

    // === 阶段 1: WM 写入 (每 WM_WRITE_INTERVAL 步) ===
    // 计算 PCA 签名, 与 50 槽位匹配, 新颖模式 LRU 替换
    if (step > PCA_WARMUP_STEPS && step % WM_WRITE_INTERVAL == 0) {
        compute_pca_signature(d_replay_sig_);
        launch_wm_write(
            buf.d_wm_slots, d_replay_sig_,
            buf.d_wm_write_cursor, step,
            WM_SLOTS, WM_NOVELTY_THRESHOLD);
    }

    // === 阶段 2: WM 维持与注入 (每步) ===
    // 清零前额叶输入缓冲, 然后调用 wm_maintain 累加注入
    if (buf.d_prefrontal_input) {
        CUDA_CHECK_2E(cudaMemsetAsync(buf.d_prefrontal_input, 0,
                                       N_PREFRONTAL_NEURONS * sizeof(float)));
        launch_wm_maintain(
            buf.d_wm_slots, buf.d_pca_W, d_pca_mean_, buf.d_prefrontal_input,
            WM_SLOTS, N_PREFRONTAL_NEURONS, NEURONS_PER_PF_GROUP,
            WM_INJECT_THRESHOLD, WM_DECAY, PCA_N_COMPONENTS,
            N_ASSOCIATION_NEURONS_2E + N_PREFRONTAL_NEURONS);
    }

    // === 阶段 3: 保留原有 p3_wm_update_kernel (兼容性, 用于 activity_drive 统计) ===
    // 注: 原 p3_wm_update_kernel 用 activity_drive 驱动 WM 槽位, 现已被
    //     launch_wm_write (PCA 签名驱动) 替代。但保留调用以维持统计计数。
    float avg_spikes = total_steps_ > 0 ? (float)total_spikes_accum_ / (float)total_steps_ : 0.0f;
    float activity_drive = stats_.total_spikes / (avg_spikes + 1.0f);
    if (activity_drive > 2.0f) activity_drive = 2.0f;
    if (activity_drive < 0.0f) activity_drive = 0.0f;
    p3_last_activity_drive_ = activity_drive;
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

    // === 阶段 1: PSW α/β 衰减 + 弱突触重置 (保留原有逻辑) ===
    const DevPhaseParams& phase = phase_table_.get_params(step);
    float evidence_threshold = phase.prune_threshold;
    float decay_factor = 0.95f;

    int blocks = (N_TOTAL_SYNAPSES_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    structural_plasticity_decay_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_synapses, buf.d_synapse_alpha, buf.d_synapse_beta,
        N_TOTAL_SYNAPSES_2E, decay_factor, evidence_threshold);

    // === 阶段 2: 结构重建 (共激活候选 → 新突触 + 弱突触修剪) ===
    // 每 STRUCTURAL_REBUILD_INTERVAL 步触发 (当前 step % 1000 == 0 已在 step() 中判定)
    if (step > 0 && step % STRUCTURAL_REBUILD_INTERVAL == 0) {
        // 读取当前 tracker 数量
        int tracker_count = 0;
        CUDA_CHECK_2E(cudaMemcpy(&tracker_count, buf.d_tracker_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
        if (tracker_count > COACT_TRACKER_SIZE) tracker_count = COACT_TRACKER_SIZE;

        if (tracker_count > 0) {
            // 调用结构重建 (阶段1: 候选生成 + 阶段2: 修剪标记)
            launch_structural_rebuild(
                buf.d_coact_trackers, tracker_count,
                d_new_synapse_pairs_, d_new_synapse_count_, d_new_modulator_scores_,
                buf.d_synapses, buf.d_csr_row_ptr,
                N_TOTAL_NEURONS_2E,
                d_prune_marks_, d_prune_count_,
                COACT_FORM_THRESHOLD, PRUNE_WEIGHT_THRESHOLD, COACT_MAX_NEW_SYNAPSES);

            // 同步并读取计数
            CUDA_CHECK_2E(cudaDeviceSynchronize());
            int new_count = 0, prune_count = 0;
            CUDA_CHECK_2E(cudaMemcpy(&new_count, d_new_synapse_count_,
                                      sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK_2E(cudaMemcpy(&prune_count, d_prune_count_,
                                      sizeof(int), cudaMemcpyDeviceToHost));

            // 读取当前突触总数
            int n_synapses_total = 0;
            CUDA_CHECK_2E(cudaMemcpy(&n_synapses_total, buf.d_csr_row_ptr + N_TOTAL_NEURONS_2E,
                                      sizeof(int), cudaMemcpyDeviceToHost));

            // 5% 阈值判定 + CSR 重建 + Task 19: 完整性校验 + 失败回滚
            // launch_csr_rebuild_with_integrity_check 内部:
            //   1. 保存旧 CSR 副本
            //   2. 调用 launch_csr_rebuild (含 5% 判定)
            //   3. 重建后启动 csr_integrity_check_kernel 校验
            //   4. 校验失败时从旧副本回滚
            int integrity_err = launch_csr_rebuild_with_integrity_check(
                buf.d_synapses, buf.d_csr_row_ptr,
                d_new_synapse_pairs_, new_count,
                d_prune_marks_, N_TOTAL_NEURONS_2E,
                n_synapses_total,
                0);  // stream = 0 (默认流)
            bool rebuilt = (integrity_err == 0);

            structural_rebuild_count_++;
            printf("[Stage2e P3-D] step=%d 结构重建 #%d: new=%d prune=%d total=%d rebuilt=%s integrity=%s\n",
                   step, structural_rebuild_count_, new_count, prune_count,
                   n_synapses_total, rebuilt ? "YES" : "NO(threshold)",
                   integrity_err == 0 ? "OK" : "FAIL(rolled-back)");
        }
    }
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
    if (!buf.d_hippo_indices || !buf.d_replay_injection || !buf.d_pca_W) return;

    // warmup: 步数 > REPLAY_WARMUP_STEPS 才触发
    if (step <= REPLAY_WARMUP_STEPS) return;

    // ==================== Task 18: 进入睡眠态 ====================
    // 保存当前 thalamic_gain (gate_mean_) 和 ach_level, 设置 is_sleeping_=true
    // 后续 step() 的 input_inject 步检查 is_sleeping_ 跳过外部字节注入
    enter_sleep_state(step);

    // 清零重放注入缓冲
    CUDA_CHECK_2E(cudaMemsetAsync(buf.d_replay_injection, 0,
                                   N_ASSOCIATION_NEURONS_2E * sizeof(float)));

    // 调用海马重放完整流程
    launch_replay_cycle(
        buf.d_hippo_indices,
        buf.d_hippo_top_k,
        buf.d_hippo_filled_count,
        buf.d_replay_injection,
        d_replay_sig_,
        buf.d_pca_W,
        d_pca_mean_,
        d_replay_recon_,
        step,
        HIPP_INDEX_SIZE,
        HIPP_REPLAY_BATCH);

    replay_cycle_count_++;
    printf("[Stage2e P4] step=%d 睡眠重放周期 #%d 完成 (重放 %d 模式)\n",
           step, replay_cycle_count_, HIPP_REPLAY_BATCH);

    // ==================== Task 18: 退出睡眠态 ====================
    // 恢复 thalamic_gain 和 ach_level, is_sleeping_=false
    exit_sleep_state(step);
}

// ==================== Task 18: 睡眠重放状态隔离实现 ====================
// enter_sleep_state: 进入睡眠态
//   1. 保存当前 gate_mean_ (host 缓存的丘脑门控均值) 到 saved_thalamic_gain_
//   2. 保存当前 stats_.ach_level (host 缓存的 ACh 水平) 到 saved_ach_level_
//   3. 设置 stats_.ach_level *= SLEEP_ACH_FACTOR (慢波睡眠 ACh 降至 ~30%)
//   4. 设置 is_sleeping_ = true (后续 input_inject / modulatory 步查询此标志)
// 生物学意义: 慢波睡眠期间丘脑门控关闭外部输入, ACh 水平降低切换到巩固模式
// P0 修复: stats_.ach_level 已在每 100 步 modulatory 更新后同步赋值 (line 608)
void BioMechanismScheduler::enter_sleep_state(int step) {
    saved_thalamic_gain_ = gate_mean_;
    saved_ach_level_     = stats_.ach_level;
    stats_.ach_level    *= SLEEP_ACH_FACTOR;
    is_sleeping_         = true;
    sleep_cycle_count_++;
    printf("[Stage2e P4] step=%d 进入睡眠重放态, 外部输入已抑制 (ach=%.4f→%.4f)\n",
           step, saved_ach_level_, stats_.ach_level);
}

// exit_sleep_state: 退出睡眠态, 恢复清醒态参数
//   1. 恢复 stats_.ach_level = saved_ach_level_
//   2. (gate_mean_ 由下一次 launch_thalamic_gate_update 自然刷新, 无需主动恢复)
//   3. 设置 is_sleeping_ = false
void BioMechanismScheduler::exit_sleep_state(int step) {
    stats_.ach_level = saved_ach_level_;
    is_sleeping_     = false;
    printf("[Stage2e P4] step=%d 退出睡眠重放态, 恢复外部输入 (ach=%.4f)\n",
           step, stats_.ach_level);
}

// ==================== Task 2: PCA 集成实现 ====================
// launch_pca_update_cpu: 每 PCA_UPDATE_INTERVAL 步在 CPU 端执行 Oja's rule 在线学习
//   1. 收集联合皮层 (前 N_ASSOCIATION_NEURONS_2E 个神经元) 发放率快照到 CPU
//      从 d_spike_flags (bool) 拷贝并转为 float (0/1 = 当前步是否发放)
//   2. 更新滑动平均发放率 h_mean_fr_ (EMA)
//   3. CPU 端 Oja 在线更新 h_pca_W_
//      中心化向量 x[i] = fr[i] - mean[i]
//      投影  proj[k] = Σ_i W[i*K+k] · x[i]
//      更新  W[i*K+k] += η · (x[i] - W[i*K+k]·proj[k]) · proj[k]
//   4. 每 PCA_SYNC_INTERVAL 步同步 h_pca_W_ 到 GPU d_pca_W
//      (同时同步 h_mean_fr_ 到 d_pca_mean_ 供 compute_pca_signature 用)
void BioMechanismScheduler::launch_pca_update_cpu(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_spike_flags || !buf.d_pca_W) return;

    const int N = N_ASSOCIATION_NEURONS_2E;   // 50,000 联合皮层神经元
    const int K = PCA_N_COMPONENTS;            // 50 主成分
    const float lr = PCA_LEARNING_RATE;        // η = 0.01

    // 1. 收集联合皮层发放率快照: d_spike_flags (bool) → h_fr_snapshot_ (float)
    //    d_spike_flags 含全部 60K 神经元, 仅拷贝前 50K (联合皮层) 部分
    CUDA_CHECK_2E(cudaMemcpy(h_spike_buf_.data(), buf.d_spike_flags,
                             N * sizeof(bool), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        h_fr_snapshot_[i] = h_spike_buf_[i] ? 1.0f : 0.0f;
    }

    // 2. 更新滑动平均发放率 h_mean_fr_ (EMA)
    //    mean = ema · mean_old + (1-ema) · fr_new
    const float ema = h_mean_fr_ema_;
    const float inv_ema = 1.0f - ema;
    for (int i = 0; i < N; ++i) {
        h_mean_fr_[i] = ema * h_mean_fr_[i] + inv_ema * h_fr_snapshot_[i];
    }

    // 3. CPU 端 Oja's rule 在线更新 h_pca_W_
    // 阶段1: 单次遍历 N, 计算所有 K 个主成分的投影 proj[k]
    //   内层 K 循环访问连续内存 W[i*K+0..K-1], cache 友好
    float proj[PCA_N_COMPONENTS];
    for (int k = 0; k < K; ++k) proj[k] = 0.0f;
    for (int i = 0; i < N; ++i) {
        float x = h_fr_snapshot_[i] - h_mean_fr_[i];   // 中心化向量
        size_t base = (size_t)i * K;
        for (int k = 0; k < K; ++k) {
            proj[k] += h_pca_W_[base + k] * x;
        }
    }
    // 阶段2: Oja 更新 W[i][k] += η · (x[i] - W[i][k]·proj[k]) · proj[k]
    //   需阶段1 全部完成 (proj[k] 依赖所有 i), 故分两趟
    for (int i = 0; i < N; ++i) {
        float x = h_fr_snapshot_[i] - h_mean_fr_[i];
        size_t base = (size_t)i * K;
        for (int k = 0; k < K; ++k) {
            float w = h_pca_W_[base + k];
            h_pca_W_[base + k] = w + lr * (x - w * proj[k]) * proj[k];
        }
    }

    // 4. 每 PCA_SYNC_INTERVAL 步同步 h_pca_W_ 到 GPU d_pca_W
    //    同时同步 h_mean_fr_ 到 d_pca_mean_ (供 compute_pca_signature / pca_back_project)
    if (step % PCA_SYNC_INTERVAL == 0) {
        CUDA_CHECK_2E(cudaMemcpy(buf.d_pca_W, h_pca_W_.data(),
                                 (size_t)N * K * sizeof(float),
                                 cudaMemcpyHostToDevice));
        CUDA_CHECK_2E(cudaMemcpy(d_pca_mean_, h_mean_fr_.data(),
                                 N * sizeof(float), cudaMemcpyHostToDevice));
    }

    pca_update_count_++;
}

// compute_pca_signature: 从当前联合皮层发放率提取 K 维 PCA 签名 (L2 归一化)
//   供海马编码和 WM 写入调用, 调用方分配 d_signature_out [PCA_N_COMPONENTS]
//   内部: d_spike_flags (bool) → d_pca_fr_ (float), 同步 h_mean_fr_ → d_pca_mean_,
//         调用 launch_pca_encode(d_pca_W, d_pca_fr_, d_pca_mean_, d_signature_out, N, K)
void BioMechanismScheduler::compute_pca_signature(float* d_signature_out) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_pca_W || !buf.d_spike_flags || !d_pca_fr_ || !d_pca_mean_ || !d_signature_out) return;

    const int N = N_ASSOCIATION_NEURONS_2E;
    const int K = PCA_N_COMPONENTS;

    // 1. 当前发放率: d_spike_flags (bool) → d_pca_fr_ (float)
    int blocks = (N + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    pca_spike_to_float_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_spike_flags, d_pca_fr_, N);

    // 2. 同步 h_mean_fr_ 到 d_pca_mean_ (确保 GPU 端 mean 是最新的)
    CUDA_CHECK_2E(cudaMemcpy(d_pca_mean_, h_mean_fr_.data(),
                             N * sizeof(float), cudaMemcpyHostToDevice));

    // 3. 调用 PCA 签名提取 kernel (输出 L2 归一化的 K 维签名)
    launch_pca_encode(buf.d_pca_W, d_pca_fr_, d_pca_mean_,
                      d_signature_out, N, K);
}

// pca_back_project: 从 K 维签名全量反投影重建 N 维联合皮层发放率
//   供睡眠重放和 WM 注入调用, 调用方分配 d_reconstructed_out [N_ASSOCIATION_NEURONS_2E]
//   内部: 调用 launch_pca_back_project(d_pca_W, d_pca_mean_, d_signature, d_reconstructed_out, N, K)
//   重建: recon[i] = mean[i] + Σ_k sig[k] · W[i][k]
void BioMechanismScheduler::pca_back_project(const float* d_signature,
                                              float* d_reconstructed_out) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_pca_W || !d_pca_mean_ || !d_signature || !d_reconstructed_out) return;

    const int N = N_ASSOCIATION_NEURONS_2E;
    const int K = PCA_N_COMPONENTS;

    // 调用 PCA 反投影 kernel: recon[i] = mean[i] + Σ_k sig[k]·W[i][k]
    launch_pca_back_project(buf.d_pca_W, d_pca_mean_, d_signature,
                            d_reconstructed_out, N, K);
}

// ==================== Task 4-5: 在线解码 step 实现 ====================
// 流程:
//   1. 每步: launch_decode_forward (前向 + softmax + argmax) → 拷贝 predicted_byte 到 host
//   2. 仅注入步 + warmup 完成 + update_weights=true:
//      a. 把 current_input_byte 推入 input_byte_history_ 环形缓冲
//      b. target_byte = 最旧字节 (K 步前注入, 网络当前活动反映其延迟效应)
//      c. launch_decode_error: 计算 error = softmax_prob - one_hot(target), 返回 loss
//      d. launch_decode_weight_update: ΔW = -η · error · spike_flags
//      e. 更新 perplexity / accuracy 统计
//   3. 每 100 个 decode_step 调用: launch_decode_weight_normalize (行 L2 归一化)
//   4. 每 PERPLEXITY_LOG_INTERVAL 个调用: 打印 perplexity + accuracy
void BioMechanismScheduler::decode_step(uint8_t current_input_byte,
                                         bool is_inject_step,
                                         bool update_weights) {
    PersistentBuffers& buf = alloc_->buffers();
    // 防御: 解码权重未分配时直接跳过 (避免 nullptr deref)
    if (!buf.d_decode_weights || !buf.d_spike_flags || !buf.d_decode_logits) return;

    // ---- 1. 前向解码 (每步) ----
    // 内部链: forward_kernel → softmax_kernel → argmax_kernel
    launch_decode_forward(buf);

    // 拷贝预测字节到 host (同步, 因为后续准确率统计需要立即使用)
    int h_pred = 0;
    CUDA_CHECK_2E(cudaMemcpy(&h_pred, buf.d_decode_predicted_byte,
                              sizeof(int), cudaMemcpyDeviceToHost));
    last_predicted_byte_ = h_pred;

    // ---- 2. 误差驱动权重更新 (仅注入步 + warmup 完成 + 允许更新) ----
    if (is_inject_step && update_weights) {
        // 推入环形缓冲 (覆盖最旧)
        input_byte_history_[input_byte_history_idx_] = current_input_byte;
        input_byte_history_idx_ = (input_byte_history_idx_ + 1) % PREDICTION_DELAY_STEPS;
        if (input_byte_history_count_ < PREDICTION_DELAY_STEPS) {
            input_byte_history_count_++;
        }

        // 仅在 warmup 完成后 (历史缓冲填满) 才执行学习
        // target = K 步前注入的字节 (环形缓冲中最旧的一个, 即将被打入写入位置)
        if (input_byte_history_count_ >= PREDICTION_DELAY_STEPS) {
            uint8_t target_byte = input_byte_history_[input_byte_history_idx_];

            // 2a. 计算误差 + loss
            float h_loss = 0.0f;
            launch_decode_error(buf, target_byte, h_loss);

            // 2b. 权重更新 (ΔW = -η · error · spike_flags)
            launch_decode_weight_update(buf);

            // 2c. 统计
            last_decode_loss_ = h_loss;
            cross_entropy_loss_accum_ += h_loss;
            loss_accum_count_++;
            predict_total_count_++;
            if (last_predicted_byte_ == (int)target_byte) {
                correct_predict_count_++;
            }
        }
    }

    // ---- 3. 行 L2 归一化 (每 100 步) ----
    decode_step_counter_++;
    if (decode_step_counter_ % 100 == 0) {
        launch_decode_weight_normalize(buf);
    }

    // ---- 4. perplexity 日志 (每 PERPLEXITY_LOG_INTERVAL 步) ----
    if (decode_step_counter_ % PERPLEXITY_LOG_INTERVAL == 0 && loss_accum_count_ > 0) {
        float avg_loss = cross_entropy_loss_accum_ / (float)loss_accum_count_;
        float perplexity = expf(avg_loss);
        float accuracy = predict_total_count_ > 0
            ? 100.0f * (float)correct_predict_count_ / (float)predict_total_count_
            : 0.0f;
        printf("[Stage2e Decode] step=%d  avg_loss=%.4f  perplexity=%.2f  "
               "accuracy=%.2f%% (%d/%d)  last_pred=%d\n",
               decode_step_counter_, avg_loss, perplexity, accuracy,
               correct_predict_count_, predict_total_count_, last_predicted_byte_);
        // 周期性重置累积器 (输出的是窗口平均值, 不是全局平均)
        cross_entropy_loss_accum_ = 0.0f;
        loss_accum_count_ = 0;
    }
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

    // Task 2: PCA 状态日志 (每 LOG_INTERVAL_2E 步输出一次)
    //   W_norm = ||W||_F = sqrt(Σ_{i,k} W[i][k]²), 反映 PCA 基矩阵整体能量
    float pca_w_sq_sum = 0.0f;
    for (size_t idx = 0; idx < h_pca_W_.size(); ++idx) {
        pca_w_sq_sum += h_pca_W_[idx] * h_pca_W_[idx];
    }
    float pca_w_norm = sqrtf(pca_w_sq_sum);
    printf("[PCA] updates=%d W_norm=%.4f\n", pca_update_count_, pca_w_norm);
}

void BioMechanismScheduler::print_phase_change(int step, DevPhase new_phase) {
    const char* phase_names[] = {"EMBRYONIC (0-5K)", "SYNAPTOGENIC (5K-200K)",
                                 "CRITICAL (200K-800K)", "PRUNING (800K-1.5M)",
                                 "MATURE (1.5M-3M)"};
    printf("\n[Stage2e P1] ===== 发育阶段切换: %s (step %d) =====\n\n",
           phase_names[static_cast<int>(new_phase)], step);
}

} // namespace stage2e
