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
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

namespace stage2e {

// P1 占位 kernel: 简单清零 (用于未实现的慢时间尺度机制)
__global__ void p1_noop_clear_float(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 0.0f;
}

// P1 占位 kernel: 统计当前步 spike 数 (用于 P1 判据)
__global__ void count_spikes_kernel(const bool* spike_flags, int n, int* out_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (spike_flags[i]) {
        atomicAdd(out_count, 1);
    }
}

BioMechanismScheduler::BioMechanismScheduler(MemoryAllocator* alloc)
    : alloc_(alloc), stats_{}, delay_ring_idx_(0),
      total_steps_(0), total_spikes_accum_(0),
      min_spikes_per_step_(INT_MAX), max_spikes_per_step_(0),
      total_burst_steps_(0) {
    printf("[Stage2e P1] 调度器初始化完成\n");
    memset(&stats_, 0, sizeof(stats_));
    // 分配 device 端 spike 计数器
    cudaMalloc(&d_spike_counter_, sizeof(int));
    cudaMemset(d_spike_counter_, 0, sizeof(int));
}

BioMechanismScheduler::~BioMechanismScheduler() {
    printf("[Stage2e P1] 调度器销毁, 共执行 %d 步, 累计脉冲 %d\n",
           total_steps_, total_spikes_accum_);
    if (d_spike_counter_) cudaFree(d_spike_counter_);
}

void BioMechanismScheduler::step(int current_step) {
    PersistentBuffers& buf = alloc_->buffers();
    const DevPhaseParams& phase = phase_table_.get_params(current_step);

    // ==================== 快时间尺度 (每步) ====================
    // v4 流水线: delay_inject → input_inject → lif_adex → synapse_nmda → stdp_dual_trace → stdp_stp → delay_dispatch

    // 1. 延迟队列注入 (清零 input_current + 注入上一轮 ring_idx 槽位的内容)
    launch_delay_inject(alloc_, delay_ring_idx_);

    // 2. 群体编码输入注入 (每 INPUT_INJECT_INTERVAL 步注入一个字节)
    if (current_step % INPUT_INJECT_INTERVAL == 0) {
        uint8_t byte = get_byte_for_step(current_step);
        launch_input_inject(alloc_, byte);
    }

    // 3. AdEx 神经元更新 (产生 spike_flags)
    launch_lif_adex(alloc_, current_step, phase);

    // 4. NMDA 受体 + 钙浓度 (消耗 spike_flags, 更新 nmda_current)
    launch_synapse_nmda(alloc_, current_step);

    // 5. STDP 双 trace + Δw (消耗 spike_flags, 更新 weight)
    launch_stdp_dual_trace(alloc_, current_step, phase.plasticity_gain);

    // 6. STP 短期可塑性 (消耗 spike_flags, 更新 resource/utilization)
    launch_stdp_stp(alloc_, current_step);

    // 7. 延迟队列分发 (扫描 spike_flags, 写入 ring[target_ring])
    launch_delay_dispatch(alloc_, current_step, delay_ring_idx_);

    // 8. 抑制性电流清零 (P1: 抑制性网络未实现, 保持 0)
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    p1_noop_clear_float<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_inhibitory_current, N_TOTAL_NEURONS_2E);

    // ==================== 中时间尺度 (每 10 步) ====================
    // P1 占位 (Phase 2 实现): camkii, eligibility, inhibitory_network
    if (current_step % 10 == 0) {
        launch_camkii_kernel(current_step);
        launch_stdp_eligibility(current_step);
        launch_inhibitory_network(current_step);
    }

    // ==================== 慢时间尺度 (每 100 步) ====================
    // P1 占位 (Phase 2-3 实现): modulatory, scaling, wm_update
    if (current_step % 100 == 0) {
        launch_modulatory(current_step);
        launch_scaling(current_step);
        launch_wm_update(current_step);
    }

    // ==================== 极慢时间尺度 (每 1000 步) ====================
    // P1 占位 (Phase 4 实现): structural_plasticity, developmental
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

    // ==================== 统计 ====================
    // 统计当前步 spike 数 (用于 P1 判据)
    cudaMemsetAsync(d_spike_counter_, 0, sizeof(int));
    count_spikes_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_spike_flags, N_TOTAL_NEURONS_2E, d_spike_counter_);

    int step_spikes = 0;
    cudaMemcpy(&step_spikes, d_spike_counter_, sizeof(int), cudaMemcpyDeviceToHost);

    total_steps_++;
    total_spikes_accum_ += step_spikes;
    stats_.total_spikes = step_spikes;
    stats_.vram_used_bytes = alloc_->vram_used();
    stats_.vram_peak_bytes = alloc_->vram_peak();

    // P1 判据: spike count 极差 + 簇状发放
    if (step_spikes < min_spikes_per_step_) min_spikes_per_step_ = step_spikes;
    if (step_spikes > max_spikes_per_step_) max_spikes_per_step_ = step_spikes;
    // 簇状发放判定: 单步 spike > 平均的 2 倍 (简化判定)
    if (total_steps_ > 100 && step_spikes > 2 * (total_spikes_accum_ / total_steps_)) {
        total_burst_steps_++;
    }

    // 日志
    if (current_step % LOG_INTERVAL_2E == 0) {
        print_step_log(current_step);
    }
}

// ==================== P1 占位 kernel 实现 (中/慢时间尺度) ====================
// 这些在 Phase 2-4 实现, P1 保持占位

void BioMechanismScheduler::launch_camkii_kernel(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_camkii_activity) return;
    // P1 占位: 保持 0 (Phase 2 实现 CaMKII 自磷酸化)
}

void BioMechanismScheduler::launch_stdp_eligibility(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_eligibility || !buf.d_eligibility_slow) return;
    // P1 占位: 保持 0 (Phase 2 实现 2 阶 eligibility)
}

void BioMechanismScheduler::launch_inhibitory_network(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_inhibitory_current) return;
    // P1 占位: 已在主循环清零, 这里无需操作 (Phase 3 实现 3 种抑制亚型)
}

void BioMechanismScheduler::launch_modulatory(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_da_concentration) return;
    // P1 占位: 调质浓度保持 0 (Phase 2 实现 DA/ACh/NE/5HT)
}

void BioMechanismScheduler::launch_scaling(int step) {
    // P1 占位 (Phase 2 实现局部突触缩放)
}

void BioMechanismScheduler::launch_wm_update(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_wm_slots) return;
    // P1 占位 (Phase 3 实现工作记忆)
}

void BioMechanismScheduler::launch_structural_plasticity(int step) {
    PersistentBuffers& buf = alloc_->buffers();
    if (!buf.d_coact_trackers) return;
    // P1 占位 (Phase 4 实现结构可塑性)
}

void BioMechanismScheduler::launch_developmental(int step) {
    // 仅打印发育阶段切换
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
    // P1 占位 (Phase 4 实现睡眠重放)
}

// ==================== 日志 ====================

void BioMechanismScheduler::print_step_log(int step) {
    const DevPhaseParams& pp = phase_table_.get_params(step);
    DevPhase ph = phase_table_.get_phase(step);
    const char* phase_names[] = {"EMBRYONIC", "SYNAPTOGENIC", "CRITICAL", "PRUNING", "MATURE"};

    printf("[Stage2e P1] step=%6d  phase=%-12s  plast_gain=%.2f  nmda_expr=%.2f  "
           "vram=%.1fMB  spikes=%d  ring=%d  burst%%=%.1f\n",
           step, phase_names[static_cast<int>(ph)],
           pp.plasticity_gain, pp.nmda_expression,
           stats_.vram_used_bytes / (1024.0 * 1024.0),
           stats_.total_spikes, delay_ring_idx_,
           total_steps_ > 0 ? (100.0f * total_burst_steps_ / total_steps_) : 0.0f);
}

void BioMechanismScheduler::print_phase_change(int step, DevPhase new_phase) {
    const char* phase_names[] = {"EMBRYONIC (0-30K)", "SYNAPTOGENIC (30K-200K)",
                                 "CRITICAL (200K-800K)", "PRUNING (800K-1.5M)",
                                 "MATURE (1.5M-3M)"};
    printf("\n[Stage2e P1] ===== 发育阶段切换: %s (step %d) =====\n\n",
           phase_names[static_cast<int>(new_phase)], step);
}

} // namespace stage2e
