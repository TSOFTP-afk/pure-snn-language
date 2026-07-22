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
#include <climits>

namespace stage2e {

// -----------------------------------------------------------------------------
// 发育阶段参数表 (v2 修复 5)
// -----------------------------------------------------------------------------
struct DevPhaseTable {
    DevPhaseParams phases[5];

    DevPhaseTable() {
        // 胚胎期 (0 - 30K)
        phases[0] = {0.0f, 0.0f, 0.0f,  0.0f, 0.1f, 1.0f, DEV_PHASE_EMBRYO_END};
        // 突触发生期 (30K - 200K)
        phases[1] = {5.0f, 0.0f, 0.01f, 0.8f, 0.5f, 1.0f, DEV_PHASE_SYNAPTO_END};
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
    float burst_ratio() const {
        return total_steps_ > 0 ? (100.0f * total_burst_steps_ / total_steps_) : 0.0f;
    }

private:
    MemoryAllocator* alloc_;
    NetworkStats2e stats_;
    DevPhaseTable phase_table_;
    int delay_ring_idx_;

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

    // device 端 spike 计数器 (用于 atomicAdd 统计)
    int* d_spike_counter_;
    int* d_single_neuron_burst_counter_;

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

    // 日志
    void print_step_log(int step);
    void print_phase_change(int step, DevPhase new_phase);
};

} // namespace stage2e

#endif // SNN_STAGE2E_SCHEDULER_CUH
