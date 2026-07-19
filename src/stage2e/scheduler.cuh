#ifndef SNN_STAGE2E_SCHEDULER_CUH
#define SNN_STAGE2E_SCHEDULER_CUH

// =============================================================================
// Stage 2e 统一调度器 (P0 骨架)
// =============================================================================
// 对应设计文档 §5.4: v4 调度器
//
// P0 阶段:
//   - 所有 kernel 为占位实现 (no-op, 仅清零 / 简单统计)
//   - 调用顺序必须与 v4 设计一致, 验证流水线正确性
//   - 每 LOG_INTERVAL_2E 步打印一次统计信息
//
// 流水线:
//   delay_dispatch → lif_adex → synapse_nmda → stdp_dual_trace → stdp_stp
//        ↓                                                            ↓
//   (按 delay 写入队列)                                   (每10步) camkii_kernel
//                                                                  ↓
//                                                          stdp_eligibility
//   (每100步) modulatory, scaling, wm_update
//   (每1000步) structural_plasticity, developmental
//   (每10000步) replay
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

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

    // 主步进函数 (P0: 占位实现, 仅做计数与日志)
    void step(int current_step);

    // 获取统计
    const NetworkStats2e& stats() const { return stats_; }
    int delay_ring_idx() const { return delay_ring_idx_; }

    // P0 阶段统计累计
    int total_steps_executed() const { return total_steps_; }
    int total_spikes_accum() const { return total_spikes_accum_; }

private:
    MemoryAllocator* alloc_;
    NetworkStats2e stats_;
    DevPhaseTable phase_table_;
    int delay_ring_idx_;

    // P0 统计累计
    int total_steps_;
    int total_spikes_accum_;

    // --- P0 占位 kernel 启动器 (实际为 no-op 或简单 cudaMemset) ---
    void launch_delay_dispatch(int step);
    void launch_lif_adex(int step);
    void launch_synapse_nmda(int step);
    void launch_stdp_dual_trace(int step);
    void launch_stdp_stp(int step);

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
