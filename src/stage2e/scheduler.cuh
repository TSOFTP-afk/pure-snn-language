#ifndef SNN_STAGE2E_SCHEDULER_CUH
#define SNN_STAGE2E_SCHEDULER_CUH

// =============================================================================
// Stage 2e 统一调度器 (P1)
// =============================================================================
// 对应设计文档 §5.4: v4 调度器
//
// P1 阶段:
//   - 快时间尺度 kernel 全部替换为真实实现 (AdEx + NMDA + STDP + STP)
//   - 中/慢时间尺度仍为占位 (留给 Phase 2-4)
//   - 群体编码输入注入
//   - spike count 统计 (P1 判据: 极差 > 100, 簇状发放出现)
//
// 流水线:
//   delay_inject → input_inject → lif_adex → synapse_nmda
//       → stdp_dual_trace → stdp_stp → delay_dispatch
//   (每10步) camkii, stdp_eligibility, inhibitory_network
//   (每100步) modulatory, scaling, wm_update
//   (每1000步) structural_plasticity, developmental
//   (每10000步) replay
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include "thalamic_gate.cuh"
#include <climits>

namespace stage2e {

// -----------------------------------------------------------------------------
// 发育阶段参数表 (v2 修复 5)
// -----------------------------------------------------------------------------
struct DevPhaseTable {
    DevPhaseParams phases[5];

    DevPhaseTable() {
        // 胚胎期 (0 - 5K) — 缩短以加速涌现验证 (原 30K)
        phases[0] = {0.0f, 0.0f, 0.0f,  0.0f, 0.1f, 1.0f, DEV_PHASE_EMBRYO_END};
        // 突触发生期 (5K - 200K) — plast_gain 5.0→1.0 防止权重快速饱和
        phases[1] = {1.0f, 0.0f, 0.01f, 0.8f, 0.5f, 1.0f, DEV_PHASE_SYNAPTO_END};
        // 关键期 (200K - 800K)
        phases[2] = {3.0f, 0.0f, 0.0f,  1.0f, 0.8f, 1.2f, DEV_PHASE_CRITICAL_END};
        // 修剪期 (800K - 1.5M)
        phases[3] = {1.0f, 0.05f, 0.0f, 0.6f, 0.3f, 1.5f, DEV_PHASE_PRUNE_END};
        // 成熟期 (1.5M - 3M)
        phases[4] = {0.3f, 0.02f, 0.0f, 0.4f, 0.2f, 2.0f, DEV_PHASE_MATURE_END};
    }

    DevPhase get_phase(int step) const {
        for (int i = 0; i < 5; ++i) {
            if (step < phases[i].end_step) return static_cast<DevPhase>(i);
        }
        return DevPhase::MATURE;
    }

    const DevPhaseParams& get_params(int step) const {
        return phases[static_cast<int>(get_phase(step))];
    }
};

// -----------------------------------------------------------------------------
// 调度器
// -----------------------------------------------------------------------------
class BioMechanismScheduler {
public:
    BioMechanismScheduler(MemoryAllocator* alloc);
    ~BioMechanismScheduler();

    // E0 消融模式: 关闭三因素调制 + CaMKII + 调质系统 (纯 STDP 基线)
    bool e0_ablation = false;

    // 主步进函数
    void step(int current_step);

    // 获取统计
    const NetworkStats2e& stats() const { return stats_; }
    int delay_ring_idx() const { return delay_ring_idx_; }

    // P1 统计
    int total_steps_executed() const { return total_steps_; }
    int total_spikes_accum() const { return total_spikes_accum_; }
    int min_spikes_per_step() const { return min_spikes_per_step_; }
    int max_spikes_per_step() const { return max_spikes_per_step_; }
    int spike_range() const {
        return max_spikes_per_step_ - min_spikes_per_step_;
    }
    int total_burst_steps() const { return total_burst_steps_; }
    int total_single_neuron_burst_spikes() const { return total_single_neuron_burst_spikes_; }
    long long arrived_events_accum() const { return arrived_events_accum_; }
    long long dispatched_events_accum() const { return dispatched_events_accum_; }
    long long dropped_events_accum() const { return dropped_events_accum_; }
    int max_delay_slot_depth() const { return max_delay_slot_depth_; }
    int p3_inhibitory_updates() const { return p3_inhibitory_updates_; }
    int p3_wm_updates() const { return p3_wm_updates_; }
    float p3_last_activity_drive() const { return p3_last_activity_drive_; }
    int p3_kwta_updates() const { return p3_kwta_updates_; }
    int p3_kwta_active_columns() const { return p3_kwta_active_columns_; }
    int p3_kwta_winner_estimate() const { return p3_kwta_winner_estimate_; }
    int p3_kwta_suppressed_estimate() const { return p3_kwta_suppressed_estimate_; }
    int p3_kwta_target_per_column() const { return p3_kwta_target_per_column_; }

    // P3-C 语义聚类评估 (silhouette + JS divergence + 柱间差异)
    int    p3_semantic_eval_updates() const { return p3_semantic_eval_updates_; }
    double p3_silhouette_score() const { return p3_silhouette_score_; }
    double p3_js_divergence_mean() const { return p3_js_divergence_mean_; }
    double p3_js_divergence_max() const { return p3_js_divergence_max_; }
    double p3_column_ratio() const { return p3_column_ratio_; }
    int    p3_semantic_eval_step() const { return p3_semantic_eval_last_step_; }

    // 丘脑-皮层门控指标 (§1.1 注意力门控)
    float gate_mean() const { return gate_mean_; }
    float gate_open_ratio() const { return gate_open_ratio_; }

    // Phase R2 模块 C (Task 6.5): L6 反馈闭环指标
    int   l6_total_spikes_last() const { return l6_total_spikes_last_; }       // 上一步 L6 spike 总数
    float l6_activity_ema_mean() const { return l6_activity_ema_mean_; }       // L6 活动 EMA 跨柱均值
    // 层间指标 (最近一次 semantic_eval)
    int    layer_eval_step() const { return p3_semantic_eval_last_step_; }
    const double* layer_activation_delay() const { return layer_activation_delay_; }  // [5]
    const float*   layer_chi2_sig_ratio() const { return layer_chi2_sig_ratio_; }     // [5]
    const double* layer_chi2_mean() const { return layer_chi2_mean_; }                // [5]

    // P3-C: 显式触发一次语义聚类评估 (短测末尾由 main.cpp 调用)
    void run_semantic_eval(int step) { launch_semantic_eval(step); }

    // 完整 checkpoint: 所有持久 GPU 状态、调度器状态和文本游标。
    // next_step 表示恢复后第一个尚未执行的绝对 step。
    int save_checkpoint(int next_step, const char* dir, uint32_t topology_seed);
    int load_checkpoint(const char* path, int* next_step, uint32_t* topology_seed);
    int prune_checkpoints(const char* dir, int keep_latest);

    float burst_ratio() const {
        return total_steps_ > 0 ? (100.0f * total_burst_steps_ / total_steps_) : 0.0f;
    }

private:
    friend struct SchedulerCheckpointAccess;
    MemoryAllocator* alloc_;
    NetworkStats2e stats_;
    DevPhaseTable phase_table_;
    int delay_ring_idx_;
    int last_phase_;

    // P1 统计
    int total_steps_;
    int total_spikes_accum_;
    int inject_spikes_accum_;     // P1 修正: 注入步累计脉冲 (用于排除 burst 误判)
    int min_spikes_per_step_;
    int max_spikes_per_step_;
    int total_burst_steps_;  // 簇状发放步数
    int total_single_neuron_burst_spikes_;
    long long arrived_events_accum_;
    long long dispatched_events_accum_;
    long long dropped_events_accum_;
    int max_delay_slot_depth_;
    int p3_inhibitory_updates_;
    int p3_wm_updates_;
    float p3_last_activity_drive_;
    int p3_kwta_updates_;
    int p3_kwta_active_columns_;
    int p3_kwta_winner_estimate_;
    int p3_kwta_suppressed_estimate_;
    int p3_kwta_target_per_column_;

    // P3-C 语义聚类评估状态
    int    p3_semantic_eval_updates_;
    int    p3_semantic_eval_last_step_;
    double p3_silhouette_score_;
    double p3_js_divergence_mean_;
    double p3_js_divergence_max_;
    double p3_column_ratio_;

    // Phase R2 模块 C (Task 6): L6 反馈闭环 + 层间指标
    // L6 spike count 设备缓冲 (50 柱 × 4B = 200B)
    int* d_l6_column_spikes_;
    // 层间字节响应 (5 层 × 256 字节 = 5KB, 用于 chi2 计算)
    int* d_layer_byte_responses_;
    // 层间 chi2 统计设备缓冲 (5×int sig + 5×int act + 5×float chi2_sum = 40B)
    int*   d_layer_sig_count_;
    int*   d_layer_act_count_;
    float* d_layer_chi2_sum_;
    // 期望比例 (256 × 4B = 1KB, host 上传到 device, 用于 chi2 kernel)
    float* d_injections_per_byte_;
    // host 端 L6 统计 (用于 print_step_log)
    int   l6_total_spikes_last_;
    float l6_activity_ema_mean_;
    // host 端层间指标缓存 (最近一次 semantic_eval 结果)
    double layer_activation_delay_[5];  // 每层平均激活延迟 (相对注入步)
    float  layer_chi2_sig_ratio_[5];     // 每层卡方显著神经元比例
    double layer_chi2_mean_[5];          // 每层卡方均值

    // device 端 spike 计数器 (用于 atomicAdd 统计)
    int* d_spike_counter_;
    int* d_single_neuron_burst_counter_;
    int* d_p3_column_spikes_;
    int* d_p3_kwta_stats_;
    int* d_p3_column_byte_responses_;  // P3-C: 50 柱 × 256 字节 = 50KB

    // 丘脑-皮层门控 (§1.1 注意力门控)
    ThalamicGateState* d_gate_states_;        // 门控状态 (N_COLUMNS_2E 个, 每柱 16B)
    unsigned int* d_byte_history_;             // 字节历史计数 (256 个 uint32, 用于 novelty)
    float* d_gate_stats_;                      // 门控统计 [gate_mean, gate_open_ratio, gate_min, gate_max]
    float gate_mean_;                          // 最近一次门控均值 (host 缓存)
    float gate_open_ratio_;                    // 最近一次门控开启比例 (host 缓存)

    // --- P1 占位 kernel 启动器 (中/慢时间尺度, Phase 2-4 实现) ---
    void launch_camkii_kernel(int step);
    void launch_stdp_eligibility(int step);
    void launch_inhibitory_network(int step);

    void launch_modulatory(int step);
    void launch_scaling(int step);
    void launch_wm_update(int step);

    void launch_structural_plasticity(int step);
    void launch_developmental(int step);

    void launch_replay(int step);

    // P3-C: 语义聚类评估 (silhouette + JS divergence + 柱间差异)
    void launch_semantic_eval(int step);

    // 日志
    void print_step_log(int step);
    void print_phase_change(int step, DevPhase new_phase);
};

} // namespace stage2e

#endif // SNN_STAGE2E_SCHEDULER_CUH
