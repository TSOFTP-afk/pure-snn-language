// =============================================================================
// Stage 2e 统一调度器实现 (P0 骨架)
// =============================================================================
// P0 阶段所有 kernel 为占位实现:
//   - 检查 GPU 缓冲指针非空
//   - 执行简单 cudaMemset 验证可写
//   - 累计 step / spike 计数
//   - 不实际模拟生物动力学 (留给 Phase 1+)
// =============================================================================

#include "scheduler.cuh"
#include <cstdio>
#include <cstring>

namespace stage2e {

// P0 占位 kernel: 简单清零 (验证 GPU 缓冲可写)
__global__ void p0_noop_clear_float(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 0.0f;
}

__global__ void p0_noop_clear_bool(bool* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = false;
}

__global__ void p0_inject_random_spikes(NeuronStateAdEx* neurons, int n, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // P0: 用确定性模式注入模拟脉冲 (step + i 奇偶)
        // 这只是验证神经元缓冲可写, 不是真实模拟
        bool fake_spike = ((step + i) % 7 == 0);
        if (fake_spike) {
            neurons[i].membrane_potential = 0.5f;  // 占位: V_norm = 0.5
            neurons[i].last_spike_time = step;
        }
    }
}

BioMechanismScheduler::BioMechanismScheduler(MemoryAllocator* alloc)
    : alloc_(alloc), stats_{}, delay_ring_idx_(0),
      total_steps_(0), total_spikes_accum_(0) {
    printf("[Stage2e P0] 调度器初始化完成\n");
    memset(&stats_, 0, sizeof(stats_));
}

BioMechanismScheduler::~BioMechanismScheduler() {
    printf("[Stage2e P0] 调度器销毁, 共执行 %d 步, 累计占位脉冲 %d\n",
           total_steps_, total_spikes_accum_);
}

void BioMechanismScheduler::step(int current_step) {
    PersistentBuffers& buf = alloc_->buffers();

    // ==================== 快时间尺度 (每步) ====================
    // v4 流水线: delay_dispatch → lif_adex → synapse_nmda → stdp_dual_trace → stdp_stp

    launch_delay_dispatch(current_step);
    launch_lif_adex(current_step);
    launch_synapse_nmda(current_step);
    launch_stdp_dual_trace(current_step);
    launch_stdp_stp(current_step);

    // ==================== 中时间尺度 (每 10 步) ====================
    if (current_step % 10 == 0) {
        // v4 流水线: camkii_kernel → stdp_eligibility → inhibitory_network
        launch_camkii_kernel(current_step);
        launch_stdp_eligibility(current_step);
        launch_inhibitory_network(current_step);
    }

    // ==================== 慢时间尺度 (每 100 步) ====================
    if (current_step % 100 == 0) {
        launch_modulatory(current_step);
        launch_scaling(current_step);
        launch_wm_update(current_step);
    }

    // ==================== 极慢时间尺度 (每 1000 步) ====================
    if (current_step % 1000 == 0) {
        launch_structural_plasticity(current_step);
        launch_developmental(current_step);
    }

    // ==================== 睡眠重放 (每 10000 步) ====================
    if (current_step % 10000 == 0) {
        launch_replay(current_step);
    }

    // v4: 延迟环形队列指针前进
    delay_ring_idx_ = (delay_ring_idx_ + 1) % DELAY_STEPS_MAX;

    // 累计统计
    total_steps_++;
    // P0: 假性统计 (实际 spike 数在 Phase 1 lif_adex 中产生)
    int fake_spikes = (N_TOTAL_NEURONS_2E / 7);  // ~7857 / step (匹配 p0_inject_random_spikes 模式)
    total_spikes_accum_ += fake_spikes;
    stats_.total_spikes = fake_spikes;
    stats_.vram_used_bytes = alloc_->vram_used();
    stats_.vram_peak_bytes = alloc_->vram_peak();

    // 日志
    if (current_step % LOG_INTERVAL_2E == 0) {
        print_step_log(current_step);
    }
}

// ==================== P0 占位 kernel 实现 ====================
// 每个 launch 函数:
//   1. 检查指针非空
//   2. 启动简单的 GPU kernel (验证可写)
//   3. 不实际模拟动力学

void BioMechanismScheduler::launch_delay_dispatch(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_delay_ring_indices || !buf.d_delay_ring_current) return;
    // P0: 清零当前 ring 槽位 (验证可写)
    int blocks = (500000 + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p0_noop_clear_float<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_delay_ring_current + delay_ring_idx_ * 500000, 500000);
}

void BioMechanismScheduler::launch_lif_adex(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_neurons) return;
    // P0: 注入占位脉冲, 验证 NeuronStateAdEx 缓冲可写
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p0_inject_random_spikes<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_neurons, N_TOTAL_NEURONS_2E, step);
}

void BioMechanismScheduler::launch_synapse_nmda(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_nmda_current) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p0_noop_clear_float<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_nmda_current, N_TOTAL_NEURONS_2E);
}

void BioMechanismScheduler::launch_stdp_dual_trace(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_stdp_x_pre_trace) return;
    // P0: 占位 - 不更新 trace
}

void BioMechanismScheduler::launch_stdp_stp(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_synapses) return;
    // P0: 占位 - 不更新 STP
}

void BioMechanismScheduler::launch_camkii_kernel(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_camkii_activity) return;
    // P0: 占位
}

void BioMechanismScheduler::launch_stdp_eligibility(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_eligibility || !buf.d_eligibility_slow) return;
    // P0: 占位
}

void BioMechanismScheduler::launch_inhibitory_network(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_inhibitory_current) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p0_noop_clear_float<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_inhibitory_current, N_TOTAL_NEURONS_2E);
}

void BioMechanismScheduler::launch_modulatory(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_da_concentration) return;
    // P0: 占位 - 调质浓度保持 0
}

void BioMechanismScheduler::launch_scaling(int step) {
    // P0: 占位
}

void BioMechanismScheduler::launch_wm_update(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_wm_slots) return;
    // P0: 占位
}

void BioMechanismScheduler::launch_structural_plasticity(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_coact_trackers) return;
    // P0: 占位
}

void BioMechanismScheduler::launch_developmental(int step) {
    // P0: 仅打印发育阶段
    static DevPhase last_phase = static_cast<DevPhase>(255);
    DevPhase cur = phase_table_.get_phase(step);
    if (cur != last_phase) {
        print_phase_change(step, cur);
        last_phase = cur;
    }
}

void BioMechanismScheduler::launch_replay(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_hippo_indices || !buf.d_replay_injection) return;
    // P0: 占位 - 不实际重放
}

// ==================== 日志 ====================

void BioMechanismScheduler::print_step_log(int step) {
    const DevPhaseParams& pp = phase_table_.get_params(step);
    DevPhase ph = phase_table_.get_phase(step);
    const char* phase_names[] = {"EMBRYONIC", "SYNAPTOGENIC", "CRITICAL", "PRUNING", "MATURE"};

    printf("[Stage2e P0] step=%6d  phase=%-12s  plast_gain=%.2f  nmda_expr=%.2f  "
           "vram=%.1fMB  fake_spikes=%d  ring_idx=%d\n",
           step, phase_names[static_cast<int>(ph)],
           pp.plasticity_gain, pp.nmda_expression,
           stats_.vram_used_bytes / (1024.0 * 1024.0),
           stats_.total_spikes, delay_ring_idx_);
}

void BioMechanismScheduler::print_phase_change(int step, DevPhase new_phase) {
    const char* phase_names[] = {"EMBRYONIC (0-30K)", "SYNAPTOGENIC (30K-200K)",
                                 "CRITICAL (200K-800K)", "PRUNING (800K-1.5M)",
                                 "MATURE (1.5M-3M)"};
    printf("\n[Stage2e P0] ===== 发育阶段切换: %s (step %d) =====\n\n",
           phase_names[static_cast<int>(new_phase)], step);
}

} // namespace stage2e
