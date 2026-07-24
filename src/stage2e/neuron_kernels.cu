// =============================================================================
// Stage 2e 神经元 kernel 实现 (P1)
// =============================================================================
// 设计要点:
//   1. delay_inject_kernel: 在 lif_adex 之前, 把上一轮写入当前 ring_idx 槽位的
//      延迟电流注入到 d_input_current (atomicAdd 到 post 神经元)
//   2. lif_adex_kernel: AdEx 神经元更新 (适应性 + 簇状发放 + 阈值动态)
//   3. delay_dispatch_kernel: 在 lif_adex 之后, 把本步产生的 spike 按 delay_steps
//      写入环形队列的 target_ring 槽位 (atomicAdd 获取位置)
//
// 计数器管理 (scheduler 内部 host 数组):
//   h_ring_counter_history[DELAY_STEPS_MAX]: 每步 dispatch 后拷贝 device 计数器回 host
//   delay_inject 读取 h_ring_counter_history[ring_idx] (上一轮写入此槽位的项数)
//
// 数据流 (每步):
//   1. cudaMemsetAsync input_current = 0
//   2. delay_inject_kernel<<<...>>>(input_current, ring[ring_idx], h_ring_counter_history[ring_idx])
//   3. lif_adex_kernel<<<...>>>(neurons, spikes, input_current, nmda, inh)
//   4. cudaMemsetAsync ring_counter = 0  (清零所有 20 个槽位的写入计数器)
//   5. delay_dispatch_kernel<<<...>>>(spikes, synapses, ring, ring_counter, ring_idx)
//   6. cudaMemcpy h_ring_counter_history ← d_ring_counter (20 个 int)
//   7. ring_idx = (ring_idx + 1) % MAX
// =============================================================================

#include "neuron_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// delay_inject_kernel: 把延迟队列电流注入到 input_current
// 每个 thread 处理一个 slot, atomicAdd 到 post 神经元的 input_current
// =============================================================================
__global__ void delay_inject_kernel(
    float* __restrict__ input_current,
    const int* __restrict__ delay_ring_indices,
    const float* __restrict__ delay_ring_current,
    const BioSynapse* __restrict__ synapses,
    int current_ring_idx,
    int slot_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slot_count) return;

    int slot_base = current_ring_idx * DELAY_RING_SLOT_CAPACITY;
    int syn_idx = delay_ring_indices[slot_base + i];
    if (syn_idx < 0) return;

    float inject = delay_ring_current[slot_base + i];
    int post = synapses[syn_idx].post_idx;

    atomicAdd(&input_current[post], inject);
}

// =============================================================================
// lif_adex_kernel: AdEx 神经元更新 (§2.1)
// =============================================================================
__global__ void lif_adex_kernel(
    NeuronStateAdEx* __restrict__ neurons,
    bool* __restrict__ spike_flags,
    const float* __restrict__ input_current,
    const float* __restrict__ nmda_current,
    const float* __restrict__ inhibitory_current,
    int* __restrict__ single_neuron_burst_counter,
    int n_neurons,
    int step,
    float plasticity_gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    NeuronStateAdEx& n = neurons[i];

    // 不应期检查
    if (n.refractory_remaining > 0) {
        n.refractory_remaining--;
        spike_flags[i] = false;
        n.fire_rate *= 0.99f;  // 衰减
        return;
    }

    // ----- 1. 综合输入电流 (漏积分器累积, P1 修正) -----
    // P1 修正: 原版每步直接用 input_current[i], 电流无法跨步累积
    //   导致注入步 dV=0.98 < 阈值 1.0, 非注入步完全静默
    // 改: 用 NeuronStateAdEx.synaptic_current 作为漏积分器
    //   τ_syn = 3 步 (衰减系数 0.7), 让电流持续 ~5 步累积 V 越过阈值
    //   公式: s(t+1) = s(t) * 0.7 + I_external * 0.3
    //   效果: 单次注入 9.0 → 5 步累积 s ≈ 6.3 → 5 步 dV 累积 ≈ 3.4 → V 越阈值
    n.synaptic_current = n.synaptic_current * 0.7f + input_current[i] * 0.3f;
    float I_total = n.synaptic_current + nmda_current[i] - inhibitory_current[i];

    // ----- 2. AdEx 动力学 (归一化版本) -----
    // τ_m * dV/dt = -V + Δ_T * exp((V - V_T)/Δ_T) - w + I
    // τ_w * dw/dt = a*V - w
    //
    // 归一化参数:
    //   V_norm ∈ [-0.5, 1.5], 静息 0, 阈值 1.0
    //   V_T_norm = 0.8 (软阈值)
    //   Δ_T_norm = 0.1 (簇状发放强度)
    //   τ_m      = 9.37 ms
    //   τ_w      = 144 ms
    //   a_norm   = 0.0133 (适应耦合)
    //   b_norm   = 0.05 (脉冲后适应跳变)
    float V = n.membrane_potential;
    float w = n.adaptive_conductance;

    // AdEx 指数项 (簇状发放关键来源)
    float exp_arg = (V - 0.8f) / 0.1f;
    if (exp_arg > 10.0f) exp_arg = 10.0f;
    if (exp_arg < -10.0f) exp_arg = -10.0f;
    float exp_term = 0.1f * expf(exp_arg);

    // dV/dt = (-V + Δ_T*exp(...) - w + I) / τ_m
    float dV = (-V + exp_term - w + I_total) * (1.0f / 9.37f);
    // dw/dt = (a*V - w) / τ_w
    float dw = (0.0133f * V - w) * ADEX_TAU_W_INV;

    V += dV;
    w += dw;

    // 电压 clamp
    if (V < -0.5f) V = -0.5f;
    if (V > 1.5f)  V = 1.5f;

    // ----- 3. 脉冲检测 (阈值动态) -----
    float theta = ADEX_V_THRESH_NORM + static_cast<float>(n.threshold_offset) / 256.0f;
    bool spike = (V >= theta);

    spike_flags[i] = spike;

    if (spike) {
        if (n.last_spike_time >= 0 && step - n.last_spike_time <= 5) {
            atomicAdd(single_neuron_burst_counter, 1);
        }
        n.last_spike_time = step;
        n.refractory_remaining = ADEX_REFRACTORY_STEPS;
        V = ADEX_V_RESET_NORM;
        w += ADEX_B_RESET;
        // Ca²⁺ 内流
        n.ca_neuron = fminf(n.ca_neuron + 0.2f, 1.0f);
        // fire_rate 滑动平均
        n.fire_rate = n.fire_rate * 0.9f + 0.1f;
        // 阈值适应 (钠通道失活)
        int new_offset = n.threshold_offset + static_cast<int>(ADEX_THETA_ADAPT_RATE * 256.0f);
        int max_offset = static_cast<int>(ADEX_THETA_MAX * 256.0f);
        if (new_offset > max_offset) new_offset = max_offset;
        n.threshold_offset = static_cast<uint16_t>(new_offset);
    } else {
        // 阈值衰减
        n.threshold_offset = static_cast<uint16_t>(
            static_cast<float>(n.threshold_offset) * ADEX_THETA_DECAY);
        // 钙衰减
        n.ca_neuron *= (1.0f - 1.0f / NMDA_CA_TAU);
        // fire_rate 衰减
        n.fire_rate *= 0.995f;
    }

    n.membrane_potential = V;
    n.adaptive_conductance = w;

    (void)plasticity_gain;  // P1 阶段 plasticity_gain 仅影响 STDP, 不影响神经元
}

// =============================================================================
// delay_dispatch_kernel: 按突触 delay_steps 把 pre 脉冲分发到环形队列
// =============================================================================
__global__ void delay_dispatch_kernel(
    const bool* __restrict__ spike_flags,
    const int* __restrict__ csr_row_ptr,
    const BioSynapse* __restrict__ synapses,
    const uint8_t* __restrict__ synapse_delay,
    int* delay_ring_indices,
    float* delay_ring_current,
    int* ring_write_counter,
    int* dispatch_counts,
    int n_neurons,
    int current_ring_idx)
{
    int pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre >= n_neurons) return;
    if (!spike_flags[pre]) return;

    int row_start = csr_row_ptr[pre];
    int row_end   = csr_row_ptr[pre + 1];

    for (int i = row_start; i < row_end; ++i) {
        int delay = static_cast<int>(synapse_delay[i]);
        if (delay < 1) delay = 1;
        if (delay >= DELAY_STEPS_MAX) delay = DELAY_STEPS_MAX - 1;

        int target_ring = (current_ring_idx + delay) % DELAY_STEPS_MAX;

        // atomicAdd 获取写入位置
        int write_pos = atomicAdd(&ring_write_counter[target_ring], 1);
        if (write_pos >= DELAY_RING_SLOT_CAPACITY) {
            atomicAdd(&dispatch_counts[1], 1);
            continue;
        }

        int slot_offset = target_ring * DELAY_RING_SLOT_CAPACITY + write_pos;
        delay_ring_indices[slot_offset] = i;
        // 注入电流 = weight * resource (STP)
        delay_ring_current[slot_offset] = synapses[i].weight * synapses[i].resource;
        atomicAdd(&dispatch_counts[0], 1);
    }
}

// =============================================================================
// Host launchers
// =============================================================================
// 全局 device 计数器缓冲 (在首次调用时分配, 生命周期 = 程序)
static int* d_ring_write_counter = nullptr;
static int* d_dispatch_counts = nullptr;
static int  h_ring_counter_history[DELAY_STEPS_MAX] = {0};
static DelayQueueRuntimeStats g_delay_stats = {0, 0, 0, 0, 0, 0, 0};

static void ensure_counter_buffer() {
    if (d_ring_write_counter == nullptr) {
        cudaMalloc(&d_ring_write_counter, DELAY_STEPS_MAX * sizeof(int));
        cudaMemset(d_ring_write_counter, 0, DELAY_STEPS_MAX * sizeof(int));
        cudaMalloc(&d_dispatch_counts, 2 * sizeof(int));
        cudaMemset(d_dispatch_counts, 0, 2 * sizeof(int));
        memset(h_ring_counter_history, 0, sizeof(h_ring_counter_history));
        memset(&g_delay_stats, 0, sizeof(g_delay_stats));
    }
}

// scheduler 在 lif_adex 之前调用:
//   1. 清零 input_current
//   2. 启动 delay_inject_kernel (用上一步保存的 h_ring_counter_history[ring_idx])
void launch_delay_inject(MemoryAllocator* alloc, int ring_idx) {
    ensure_counter_buffer();
    PersistentBuffers& b = alloc->buffers();

    // 清零 input_current (准备接收延迟注入 + 后续外部输入)
    cudaMemsetAsync(b.d_input_current, 0, N_TOTAL_NEURONS_2E * sizeof(float));

    int slot_count = h_ring_counter_history[ring_idx];
    if (slot_count > DELAY_RING_SLOT_CAPACITY) slot_count = DELAY_RING_SLOT_CAPACITY;

    g_delay_stats.last_arrived_events = slot_count > 0 ? slot_count : 0;
    g_delay_stats.arrived_events += g_delay_stats.last_arrived_events;

    if (slot_count > 0) {
        int blocks = (slot_count + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        delay_inject_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
            b.d_input_current,
            b.d_delay_ring_indices,
            b.d_delay_ring_current,
            b.d_synapses,
            ring_idx,
            slot_count);
    }

    h_ring_counter_history[ring_idx] = 0;
    cudaMemsetAsync(d_ring_write_counter + ring_idx, 0, sizeof(int));
}

// scheduler 在 lif_adex 之后调用:
//   1. 清零 d_ring_write_counter
//   2. 启动 delay_dispatch_kernel (写入 target_ring = (ring_idx + delay) % MAX)
//   3. 同步并拷贝 d_ring_write_counter 到 h_ring_counter_history
void launch_delay_dispatch(MemoryAllocator* alloc, int step, int ring_idx) {
    ensure_counter_buffer();
    PersistentBuffers& b = alloc->buffers();

    cudaMemsetAsync(d_dispatch_counts, 0, 2 * sizeof(int));

    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    delay_dispatch_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_spike_flags,
        b.d_csr_row_ptr,
        b.d_synapses,
        b.d_synapse_delay,
        b.d_delay_ring_indices,
        b.d_delay_ring_current,
        d_ring_write_counter,
        d_dispatch_counts,
        N_TOTAL_NEURONS_2E,
        ring_idx);

    // 同步 + 拷贝计数器回 host
    cudaDeviceSynchronize();
    cudaMemcpy(h_ring_counter_history, d_ring_write_counter,
               DELAY_STEPS_MAX * sizeof(int), cudaMemcpyDeviceToHost);

    int h_dispatch_counts[2] = {0, 0};
    cudaMemcpy(h_dispatch_counts, d_dispatch_counts,
               2 * sizeof(int), cudaMemcpyDeviceToHost);
    g_delay_stats.last_dispatched_events = h_dispatch_counts[0];
    g_delay_stats.last_dropped_events = h_dispatch_counts[1];
    g_delay_stats.dispatched_events += h_dispatch_counts[0];
    g_delay_stats.dropped_events += h_dispatch_counts[1];
    for (int i = 0; i < DELAY_STEPS_MAX; ++i) {
        if (h_ring_counter_history[i] > g_delay_stats.max_slot_depth) {
            g_delay_stats.max_slot_depth = h_ring_counter_history[i];
        }
    }
}

void launch_lif_adex(MemoryAllocator* alloc, int step, const DevPhaseParams& phase,
                     int* d_single_neuron_burst_counter) {
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    lif_adex_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_neurons,
        b.d_spike_flags,
        b.d_input_current,
        b.d_nmda_current,
        b.d_inhibitory_current,
        d_single_neuron_burst_counter,
        N_TOTAL_NEURONS_2E,
        step,
        phase.plasticity_gain);
}

int delay_queue_last_arrived_events() {
    return g_delay_stats.last_arrived_events;
}

const DelayQueueRuntimeStats& delay_queue_stats() {
    return g_delay_stats;
}

bool export_delay_queue_state(DelayQueueCheckpointState* state) {
    if (!state) return false;
    ensure_counter_buffer();
    cudaError_t err = cudaMemcpy(state->ring_counter_history, d_ring_write_counter,
                                 sizeof(state->ring_counter_history), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return false;
    state->stats = g_delay_stats;
    return true;
}

bool import_delay_queue_state(const DelayQueueCheckpointState& state) {
    ensure_counter_buffer();
    memcpy(h_ring_counter_history, state.ring_counter_history,
           sizeof(h_ring_counter_history));
    g_delay_stats = state.stats;
    return cudaMemcpy(d_ring_write_counter, state.ring_counter_history,
                      sizeof(state.ring_counter_history), cudaMemcpyHostToDevice) == cudaSuccess;
}

} // namespace stage2e
