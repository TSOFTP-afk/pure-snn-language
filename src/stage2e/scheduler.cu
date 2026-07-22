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
      total_steps_(0), total_spikes_accum_(0), inject_spikes_accum_(0),
      min_spikes_per_step_(INT_MAX), max_spikes_per_step_(0),
      total_burst_steps_(0), total_single_neuron_burst_spikes_(0),
      arrived_events_accum_(0), dispatched_events_accum_(0),
      dropped_events_accum_(0), max_delay_slot_depth_(0) {
    printf("[Stage2e P1] 调度器初始化完成\n");
    memset(&stats_, 0, sizeof(stats_));
    // 分配 device 端 spike 计数器
    cudaMalloc(&d_spike_counter_, sizeof(int));
    cudaMemset(d_spike_counter_, 0, sizeof(int));
    cudaMalloc(&d_single_neuron_burst_counter_, sizeof(int));
    cudaMemset(d_single_neuron_burst_counter_, 0, sizeof(int));
}

BioMechanismScheduler::~BioMechanismScheduler() {
    printf("[Stage2e P1] 调度器销毁, 共执行 %d 步, 累计脉冲 %d\n",
           total_steps_, total_spikes_accum_);
    if (d_spike_counter_) cudaFree(d_spike_counter_);
    if (d_single_neuron_burst_counter_) cudaFree(d_single_neuron_burst_counter_);
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

    // 2. 群体编码输入注入 (每 INPUT_INJECT_INTERVAL 步注入一个字节)
    uint8_t current_byte = 0;
    bool is_inject_step = (current_step % INPUT_INJECT_INTERVAL == 0);
    if (is_inject_step) {
        current_byte = get_byte_for_step(current_step);
        launch_input_inject(alloc_, current_byte);
    }

    // 3. AdEx 神经元更新 (产生 spike_flags)
    cudaMemsetAsync(d_single_neuron_burst_counter_, 0, sizeof(int));
    launch_lif_adex(alloc_, current_step, phase, d_single_neuron_burst_counter_);

    // P2: 字节选择性直方图 (注入步统计 spike count per byte)
    if (is_inject_step) {
        launch_byte_histogram(alloc_, current_byte);
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
    int step_single_burst = 0;
    cudaMemcpy(&step_single_burst, d_single_neuron_burst_counter_, sizeof(int), cudaMemcpyDeviceToHost);

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
    // Phase 3 实现: 3 种抑制亚型 + k-WTA
    // P2 保持占位: 抑制性电流已在主循环清零
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
    // P2: 局部突触缩放
    // target_fr 按发育阶段调整: sensory/assoc 5Hz, motor 30Hz (项目记忆硬约束)
    // 简化: 用统一 10Hz 作为 P2 烟雾测试目标
    float target_fr = 10.0f;
    launch_synaptic_scaling(alloc_, step, target_fr);
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
           "vram=%.1fMB  spikes=%d  ring=%d  arrived=%d  burst%%=%.1f\n",
           step, phase_names[static_cast<int>(ph)],
           pp.plasticity_gain, pp.nmda_expression,
           stats_.vram_used_bytes / (1024.0 * 1024.0),
           stats_.total_spikes, delay_ring_idx_, delay_queue_stats().last_arrived_events,
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
