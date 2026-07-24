// =============================================================================
// Stage 2e 丘脑-皮层门控实现 (§1.1 注意力门控 + Phase R2 模块 C: L6 反馈闭环)
// =============================================================================
// 四个 kernel:
//   1. init_thalamic_gate_kernel: 每柱 gate_signal=GATE_INITIAL_SIGNAL, ema=0
//   2. thalamic_gate_update_kernel: 每柱一个 thread, 更新门控状态
//      - 读取 current_spikes = d_column_spikes[col] (全柱)
//      - 读取 l6_current = d_l6_column_spikes[col] (仅 L6 层, region==3)
//      - 计算 activity_norm = clamp((activity_ema - current_spikes) / max(activity_ema, 1.0f), -1, 1)
//      - 计算 l6_norm = clamp((l6_activity_ema - l6_current) / max(l6_activity_ema, 1.0f), -1, 1)
//        (高 L6 活动 → l6_norm < 0 → 门控关闭; 低 L6 活动 → l6_norm > 0 → 门控打开)
//      - 若 is_inject_step: 计算 novelty = 1.0f - (byte_history[current_byte] / max(total_history, 1))
//        并 atomicAdd 更新 byte_history[current_byte]
//      - 更新 activity_ema = decay * activity_ema + (1-decay) * current_spikes
//      - 更新 l6_activity_ema = decay * l6_activity_ema + (1-decay) * l6_current
//      - 更新 novelty_ema  = decay * novelty_ema  + (1-decay) * novelty
//      - 计算 gate_target = 0.5 + COUP_ACTIVITY * activity_norm
//                                 + COUP_NOVELTY * novelty_ema
//                                 + COUP_L6_FEEDBACK * l6_norm
//      - clamp gate_target 到 [GATE_MIN, GATE_MAX]
//      - gate_signal += UPDATE_RATE * (gate_target - gate_signal)
//   3. thalamic_gate_stats_kernel: 统计 gate_mean / gate_open_ratio / gate_min / gate_max
//      用 shared memory 归约 (50 柱很少, 单 block 即可)
// =============================================================================

#include "thalamic_gate.cuh"
#include <cstdio>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// 1. 初始化 kernel
// =============================================================================
__global__ void init_thalamic_gate_kernel(ThalamicGateState* d_gate_states, int n_columns) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns) return;
    d_gate_states[col].gate_signal     = GATE_INITIAL_SIGNAL;  // 半开, 中性
    d_gate_states[col].activity_ema    = 0.0f;
    d_gate_states[col].novelty_ema     = 0.0f;
    d_gate_states[col].l6_activity_ema = 0.0f;  // L6 活动 EMA 初始 0
}

// =============================================================================
// 2. 门控更新 kernel (每柱一个 thread)
// =============================================================================
// 注意:
//   - byte_history 的更新用 atomicAdd (多柱可能同时更新同一字节计数)
//   - novelty 读取用当前值即可 (容忍 1 步延迟)
//   - total_history 在 kernel 内动态求和 (256 个元素, 单 thread 串行求和很快)
//   - 非注入步 novelty=0 (无新字节)
// =============================================================================
__global__ void thalamic_gate_update_kernel(
    ThalamicGateState* __restrict__ d_gate_states,
    const int* __restrict__ d_column_spikes,
    const int* __restrict__ d_l6_column_spikes,
    uint8_t current_byte,
    int is_inject_step,
    unsigned int* __restrict__ d_byte_history,
    float* __restrict__ d_gate_stats,
    int n_columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns) return;

    ThalamicGateState& g = d_gate_states[col];

    // 1. 读取当前柱 spike count (全柱统计)
    float current_spikes = static_cast<float>(d_column_spikes[col]);

    // 2. 计算 activity_norm = clamp((activity_ema - current_spikes) / max(activity_ema, 1.0f), -1.0f, 1.0f)
    //    活动低于 EMA → activity_norm > 0 (门控开大, 补偿)
    //    活动高于 EMA → activity_norm < 0 (门控关小, 保护)
    float act_denom = g.activity_ema > 1.0f ? g.activity_ema : 1.0f;
    float activity_norm = (g.activity_ema - current_spikes) / act_denom;
    // clamp 到 [-1, 1] (用 fminf/fmaxf, 避免 CUDA clamp 依赖)
    activity_norm = fminf(1.0f, fmaxf(-1.0f, activity_norm));

    // 3. L6 反馈闭环 (Phase R2 模块 C: 皮层-丘脑闭环)
    //    读取 L6 层 spike count (仅 region==3 神经元, 由 scheduler 预先统计)
    //    l6_norm = (l6_ema - l6_current) / max(l6_ema, 1.0f)
    //      - 高 L6 活动 (current > ema) → l6_norm < 0 → gate_target 减小 (门控关闭, 皮层"已收到")
    //      - 低 L6 活动 (current < ema) → l6_norm > 0 → gate_target 增大 (门控打开, 皮层"还要输入")
    //    符号方向与 activity_norm 一致 (+COUP * norm), 符合预测编码逻辑
    float l6_current = static_cast<float>(d_l6_column_spikes[col]);
    float l6_denom = g.l6_activity_ema > 1.0f ? g.l6_activity_ema : 1.0f;
    float l6_norm = (g.l6_activity_ema - l6_current) / l6_denom;
    l6_norm = fminf(1.0f, fmaxf(-1.0f, l6_norm));

    // 4. 计算 novelty (仅注入步)
    float novelty = 0.0f;
    if (is_inject_step) {
        // 动态求和 total_history (256 个元素, 串行求和, 单 thread 仅 ~256 次加法)
        unsigned int total_history = 0u;
        for (int b = 0; b < 256; ++b) {
            total_history += d_byte_history[b];
        }
        // novelty = 1 - (当前字节历史计数 / 总历史计数)
        // 首次出现的字节: byte_history[current_byte]=0 → novelty≈1.0
        // 频繁出现的字节: byte_history[current_byte] 大 → novelty≈0
        unsigned int byte_count = d_byte_history[current_byte];
        unsigned int total_max = total_history > 1u ? total_history : 1u;
        novelty = 1.0f - static_cast<float>(byte_count) / static_cast<float>(total_max);
        // clamp 到 [0, 1]
        novelty = fminf(1.0f, fmaxf(0.0f, novelty));

        // 更新 byte_history (atomicAdd, 多柱并发更新同一字节计数)
        // 注意: 所有柱读到的是更新前的值, 但 novelty 计算用更新前的值即可 (容忍 1 步延迟)
        atomicAdd(&d_byte_history[current_byte], 1u);
    }

    // 5. 更新 activity_ema (慢速 EMA)
    g.activity_ema = GATE_ACTIVITY_EMA_DECAY * g.activity_ema
                   + (1.0f - GATE_ACTIVITY_EMA_DECAY) * current_spikes;

    // 6. 更新 l6_activity_ema (慢速 EMA, 复用 activity EMA 衰减率)
    g.l6_activity_ema = GATE_ACTIVITY_EMA_DECAY * g.l6_activity_ema
                      + (1.0f - GATE_ACTIVITY_EMA_DECAY) * l6_current;

    // 7. 更新 novelty_ema (慢速 EMA)
    //    非注入步 novelty=0, novelty_ema 会缓慢回归 0
    g.novelty_ema = GATE_NOVELTY_EMA_DECAY * g.novelty_ema
                  + (1.0f - GATE_NOVELTY_EMA_DECAY) * novelty;

    // 8. 计算 gate_target (预测编码式闭环)
    //    0.5 = 中性门控 (半开)
    //    + activity_norm * COUP: 活动补偿 (过低→开大, 过高→关小)
    //    + novelty_ema * COUP: novelty 增强 (新输入→开大)
    //    + l6_norm * COUP: L6 反馈 (高 L6 活动→关小, 低 L6 活动→开大)
    float gate_target = 0.5f
                      + GATE_ACTIVITY_COUP * activity_norm
                      + GATE_NOVELTY_COUP * g.novelty_ema
                      + GATE_L6_FEEDBACK_COUP * l6_norm;

    // 9. clamp gate_target 到 [GATE_MIN, GATE_MAX]
    gate_target = fminf(GATE_MAX, fmaxf(GATE_MIN, gate_target));

    // 10. 慢速更新 gate_signal (避免活动剧烈波动)
    g.gate_signal += GATE_UPDATE_RATE * (gate_target - g.gate_signal);
    // 再次 clamp (确保数值稳定)
    g.gate_signal = fminf(1.0f, fmaxf(0.0f, g.gate_signal));
}

// =============================================================================
// 3. 统计 kernel (单 block, shared memory 归约)
// =============================================================================
// 输出: d_gate_stats[4] = [gate_mean, gate_open_ratio, gate_min, gate_max]
// gate_open_ratio: gate_signal > 0.5 的柱比例
// 50 柱很少, 用单 block + shared memory 归约
// =============================================================================
__global__ void thalamic_gate_stats_kernel(
    const ThalamicGateState* __restrict__ d_gate_states,
    float* __restrict__ d_gate_stats,
    int n_columns)
{
    // 用 shared memory 累积 (简单实现: 单 thread 串行归约, 50 柱很少)
    // 这里用第一个 thread 做归约, 其他 thread 空闲 (50 柱开销可忽略)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        int open_count = 0;
        float gmin = d_gate_states[0].gate_signal;
        float gmax = d_gate_states[0].gate_signal;

        for (int i = 0; i < n_columns; ++i) {
            float g = d_gate_states[i].gate_signal;
            sum += g;
            if (g > 0.5f) open_count++;
            if (g < gmin) gmin = g;
            if (g > gmax) gmax = g;
        }

        float mean = n_columns > 0 ? sum / static_cast<float>(n_columns) : 0.0f;
        float open_ratio = n_columns > 0 ? static_cast<float>(open_count) / static_cast<float>(n_columns) : 0.0f;

        d_gate_stats[0] = mean;
        d_gate_stats[1] = open_ratio;
        d_gate_stats[2] = gmin;
        d_gate_stats[3] = gmax;
    }
}

// =============================================================================
// Host launcher: 初始化
// =============================================================================
void init_thalamic_gate(ThalamicGateState* d_gate_states) {
    // 50 柱 → 1 block (256 threads) 足够
    int threads = (N_COLUMNS_2E <= 256) ? N_COLUMNS_2E : 256;
    init_thalamic_gate_kernel<<<1, threads>>>(d_gate_states, N_COLUMNS_2E);
}

// =============================================================================
// Host launcher: 每步更新门控状态 + 统计
// =============================================================================
void launch_thalamic_gate_update(ThalamicGateState* d_gate_states,
                                  const int* d_column_spikes,
                                  const int* d_l6_column_spikes,
                                  uint8_t current_byte,
                                  bool is_inject_step,
                                  unsigned int* d_byte_history,
                                  float* d_gate_stats)
{
    // 50 柱 → 1 block (256 threads) 足够
    int threads = (N_COLUMNS_2E <= 256) ? N_COLUMNS_2E : 256;

    // 门控状态更新 (含 L6 反馈闭环)
    thalamic_gate_update_kernel<<<1, threads>>>(
        d_gate_states,
        d_column_spikes,
        d_l6_column_spikes,
        current_byte,
        is_inject_step ? 1 : 0,
        d_byte_history,
        d_gate_stats,
        N_COLUMNS_2E);

    // 统计 (单 block, 单 thread 串行归约 50 柱)
    thalamic_gate_stats_kernel<<<1, 1>>>(
        d_gate_states,
        d_gate_stats,
        N_COLUMNS_2E);
}

} // namespace stage2e
