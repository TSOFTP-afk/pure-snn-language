// =============================================================================
// Stage 2e 调质系统 + DA价值函数 + 字节选择性统计 实现 (P2)
// =============================================================================
// 设计要点:
//   1. modulatory_kernel: 4种调质浓度衰减 + 信号驱动
//   2. da_value_function: 亚柱级 V(s) = w_value · φ(s), TD学习
//   3. byte_histogram: 每注入步统计 spike count per byte
// =============================================================================

#include "modulatory_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

namespace stage2e {

// ==================== GPU kernel ====================

// 调质浓度衰减 + 信号驱动 (每100步调用, 但浓度本身每步衰减)
// 每 thread 处理一个神经元
__global__ void modulatory_kernel(
    float* __restrict__ da_conc,
    float* __restrict__ ach_conc,
    float* __restrict__ ne_conc,
    float* __restrict__ ht5_conc,
    int n_neurons,
    float da_signal,       // DA 信号 (δ + 基线)
    float ach_signal,      // ACh 信号 (惊奇 + 注意力)
    float ne_signal,       // NE 信号 (KL散度触发)
    float ht5_signal,      // 5HT 信号 (预测误差持续负)
    float da_decay,        // DA 衰减率 (exp(-100/tau))
    float ach_decay,
    float ne_decay,
    float ht5_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    // 衰减
    da_conc[i]  *= da_decay;
    ach_conc[i] *= ach_decay;
    ne_conc[i]  *= ne_decay;
    ht5_conc[i] *= ht5_decay;

    // 信号驱动 (加到所有神经元, 后续突触级受体差异化响应)
    da_conc[i]  += da_signal;
    ach_conc[i] += ach_signal;
    ne_conc[i]  += ne_signal;
    ht5_conc[i] += ht5_signal;

    // clamp
    if (da_conc[i]  < 0.0f) da_conc[i]  = 0.0f;
    if (da_conc[i]  > 2.0f) da_conc[i]  = 2.0f;
    if (ach_conc[i] < 0.0f) ach_conc[i] = 0.0f;
    if (ach_conc[i] > 2.0f) ach_conc[i] = 2.0f;
    if (ne_conc[i]  < 0.0f) ne_conc[i]  = 0.0f;
    if (ne_conc[i]  > 2.0f) ne_conc[i]  = 2.0f;
    if (ht5_conc[i] < 0.0f) ht5_conc[i] = 0.0f;
    if (ht5_conc[i] > 2.0f) ht5_conc[i] = 2.0f;
}

// 亚柱发放直方图计算 (从 spike_flags 聚合到 200 维亚柱级)
// 每 thread 处理一个亚柱 (50柱 × 4亚柱 = 200)
__global__ void subcolumn_fr_kernel(
    const bool* __restrict__ spike_flags,
    float* __restrict__ subcolumn_fr,
    int n_neurons,
    int neurons_per_subcolumn)
{
    int sc = blockIdx.x * blockDim.x + threadIdx.x;
    if (sc >= W_VALUE_DIM) return;

    int start = sc * neurons_per_subcolumn;
    int end = start + neurons_per_subcolumn;
    if (end > n_neurons) end = n_neurons;

    int count = 0;
    for (int i = start; i < end; ++i) {
        if (spike_flags[i]) count++;
    }
    // 发放率 = spike count / 神经元数
    subcolumn_fr[sc] = static_cast<float>(count) / neurons_per_subcolumn;
}

// V(s) = w_value · φ(s)  (线性价值函数)
// φ(s) = subcolumn_fr (200维)
__global__ void value_function_kernel(
    const float* __restrict__ subcolumn_fr,
    const float* __restrict__ w_value,
    float* __restrict__ v_out,
    int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        float v = 0.0f;
        for (int k = 0; k < dim; ++k) {
            v += w_value[k] * subcolumn_fr[k];
        }
        *v_out = v;
    }
}

// TD学习: w_value += η · δ · φ(s)
// w_pred 更新: w_pred += η_pred · δ_pred · φ(s)
__global__ void td_update_kernel(
    float* __restrict__ w_value,
    float* __restrict__ w_pred,
    const float* __restrict__ subcolumn_fr,
    const float* __restrict__ baseline_fr,
    float delta,
    int dim)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;

    // w_value 更新 (TD error 驱动)
    w_value[k] += ETA_VALUE * delta * subcolumn_fr[k];
    // clamp 防发散
    if (w_value[k] > 1.0f) w_value[k] = 1.0f;
    if (w_value[k] < -1.0f) w_value[k] = -1.0f;

    // w_pred 更新 (预测器: 预测下一步 fr)
    // 简化: w_pred 对角项增强
    float pred_error = subcolumn_fr[k] - baseline_fr[k];
    w_pred[k * dim + k] += ETA_PRED * pred_error * subcolumn_fr[k];
}

// EMA 基线更新 + 预测 fr
__global__ void baseline_update_kernel(
    float* __restrict__ baseline_fr,
    float* __restrict__ pred_fr,
    const float* __restrict__ subcolumn_fr,
    const float* __restrict__ w_pred,
    int dim)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;

    // EMA 基线
    baseline_fr[k] = NOVELTY_EMA_BETA * baseline_fr[k]
                   + (1.0f - NOVELTY_EMA_BETA) * subcolumn_fr[k];

    // 预测 fr (简化: 仅对角项)
    pred_fr[k] = w_pred[k * dim + k] * subcolumn_fr[k];
}

// 字节选择性直方图 (每注入步)
// 统计当前字节对应的总 spike 数
__global__ void byte_histogram_kernel(
    const bool* __restrict__ spike_flags,
    int* __restrict__ byte_histogram,
    int n_neurons,
    int current_byte)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    if (spike_flags[i]) {
        atomicAdd(&byte_histogram[current_byte], 1);
    }
}

// ==================== Host 端 launch ====================

static float h_v_s = 0.0f;
static float h_v_sp = 0.0f;
static float* d_v_scratch = nullptr;

ModulatoryRuntimeState export_modulatory_runtime_state() {
    return {h_v_s, h_v_sp};
}

void import_modulatory_runtime_state(const ModulatoryRuntimeState& state) {
    h_v_s = state.v_s;
    h_v_sp = state.v_sp;
}

static void ensure_v_scratch() {
    if (d_v_scratch == nullptr) {
        cudaMalloc(&d_v_scratch, sizeof(float));
        cudaMemset(d_v_scratch, 0, sizeof(float));
    }
}

void launch_modulatory(MemoryAllocator* alloc, int step,
                       float reward_signal, float novelty,
                       float pred_succ, float kl_divergence,
                       float da_delta)
{
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;

    // 调质信号计算
    // DA: δ(t) 驱动, 基线 0.1
    float da_signal = 0.1f + (da_delta > 0 ? da_delta : 0.3f * da_delta);
    if (da_signal < 0.0f) da_signal = 0.0f;

    // ACh: 基线 0.2 + 惊奇 (novelty) + 注意力 (gamma sync 简化为 pred_succ)
    float ach_signal = 0.2f + 0.3f * novelty + 0.1f * pred_succ;

    // NE: 基线 0.05 + KL 散度触发
    float ne_signal = 0.05f;
    if (kl_divergence > 0.5f) {
        ne_signal += 0.5f * kl_divergence;
    }

    // 5HT: 基线 0.1 + 预测误差持续负时上升
    float ht5_signal = 0.1f;
    if (da_delta < -0.5f) {
        ht5_signal += 0.3f * fabsf(da_delta);
    }

    // 衰减率: 100步 / tau
    float da_decay  = expf(-100.0f / DA_TAU);
    float ach_decay = expf(-100.0f / ACH_TAU);
    float ne_decay  = expf(-100.0f / NE_TAU);
    float ht5_decay = expf(-100.0f / HT5_TAU);

    modulatory_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_da_concentration, b.d_ach_concentration,
        b.d_ne_concentration, b.d_ht5_concentration,
        N_TOTAL_NEURONS_2E,
        da_signal, ach_signal, ne_signal, ht5_signal,
        da_decay, ach_decay, ne_decay, ht5_decay);
}

void launch_da_value_function(MemoryAllocator* alloc, int step,
                              float reward, float* out_v_s, float* out_v_sp)
{
    PersistentBuffers& b = alloc->buffers();
    ensure_v_scratch();

    // 1. 计算亚柱发放直方图 (200维)
    int neurons_per_sc = N_TOTAL_NEURONS_2E / W_VALUE_DIM;
    int sc_blocks = (W_VALUE_DIM + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    subcolumn_fr_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_spike_flags, b.d_subcolumn_fr, N_TOTAL_NEURONS_2E, neurons_per_sc);

    // 2. 计算 V(s) (当前)
    value_function_kernel<<<1, 1>>>(
        b.d_subcolumn_fr, b.d_w_value, d_v_scratch, W_VALUE_DIM);
    cudaMemcpy(&h_v_s, d_v_scratch, sizeof(float), cudaMemcpyDeviceToHost);

    // 3. EMA 基线更新 + 预测 fr (得到 V(s') 的近似)
    baseline_update_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_baseline_fr, b.d_pred_fr, b.d_subcolumn_fr, b.d_w_pred, W_VALUE_DIM);

    // 4. 计算 V(s') (用预测的 fr)
    value_function_kernel<<<1, 1>>>(
        b.d_pred_fr, b.d_w_value, d_v_scratch, W_VALUE_DIM);
    cudaMemcpy(&h_v_sp, d_v_scratch, sizeof(float), cudaMemcpyDeviceToHost);

    // 5. TD error: δ = R + γ·V(s') - V(s)
    float delta = reward + TD_GAMMA * h_v_sp - h_v_s;

    // 6. TD 学习更新
    td_update_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_w_value, b.d_w_pred, b.d_subcolumn_fr, b.d_baseline_fr,
        delta, W_VALUE_DIM);

    if (out_v_s)  *out_v_s  = h_v_s;
    if (out_v_sp) *out_v_sp = h_v_sp;
}

void launch_byte_histogram(MemoryAllocator* alloc, uint8_t current_byte)
{
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    byte_histogram_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_spike_flags, b.d_byte_histogram, N_TOTAL_NEURONS_2E, current_byte);
}

void get_byte_histogram(MemoryAllocator* alloc, int* out_hist)
{
    PersistentBuffers& b = alloc->buffers();
    cudaMemcpy(out_hist, b.d_byte_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
}

ModulatoryStats get_modulatory_stats(MemoryAllocator* alloc)
{
    PersistentBuffers& b = alloc->buffers();
    ModulatoryStats stats = {};

    // 采样前 1000 个神经元的调质浓度均值
    const int sample = 1000;
    float h_da[sample], h_ach[sample], h_ne[sample], h_ht5[sample];
    cudaMemcpy(h_da,  b.d_da_concentration,  sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ach, b.d_ach_concentration, sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ne,  b.d_ne_concentration,  sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ht5, b.d_ht5_concentration, sample * sizeof(float), cudaMemcpyDeviceToHost);

    double s_da = 0, s_ach = 0, s_ne = 0, s_ht5 = 0;
    for (int i = 0; i < sample; ++i) {
        s_da  += h_da[i];
        s_ach += h_ach[i];
        s_ne  += h_ne[i];
        s_ht5 += h_ht5[i];
    }
    stats.da_mean  = static_cast<float>(s_da  / sample);
    stats.ach_mean = static_cast<float>(s_ach / sample);
    stats.ne_mean  = static_cast<float>(s_ne  / sample);
    stats.ht5_mean = static_cast<float>(s_ht5 / sample);
    stats.v_s  = h_v_s;
    stats.v_sp = h_v_sp;
    return stats;
}

} // namespace stage2e
