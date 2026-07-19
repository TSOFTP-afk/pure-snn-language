#ifndef SNN_CONFIG_H
#define SNN_CONFIG_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define N_SENSORY_NEURONS     2000
#define N_ASSOCIATION_NEURONS 6000
#define N_MOTOR_NEURONS       2000
#define N_TOTAL_NEURONS       (N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS + N_MOTOR_NEURONS)
#define SYNAPSES_PER_NEURON   100
#define N_TOTAL_SYNAPSES      (N_TOTAL_NEURONS * SYNAPSES_PER_NEURON)
#define EXCITATORY_RATIO      0.8f

#define LIF_BETA              0.95f
#define LIF_THRESHOLD         1.0f
#define LIF_RESET             0.0f
#define LIF_REST              0.0f
#define LIF_REFRACTORY        2

// STDP 学习率（提高 5× 加速学习，配合 dopamine 调制效果更明显）
#define STDP_A_PLUS           0.05f
#define STDP_A_MINUS          0.05f
#define STDP_TAU_PLUS         20.0f
#define STDP_TAU_MINUS        20.0f
#define STDP_W_MIN            0.0f
#define STDP_W_MAX            1.0f

// Homeostatic plasticity（Intrinsic Plasticity）
// 每个神经元通过调节阈值维持目标发放率，避免群体活动塌缩到 0 或癫痫
// 不同脑区使用不同目标率（方案 B）：
//   - 感觉/联合皮层：低目标率（5Hz），保持稀疏编码
//   - 运动皮层：高目标率（20Hz），确保有输出活动
#define HOMEOSTATIC_TARGET_RATE_SENSORY     0.005f   // 5Hz
#define HOMEOSTATIC_TARGET_RATE_ASSOCIATION 0.005f   // 5Hz
#define HOMEOSTATIC_TARGET_RATE_MOTOR       0.030f   // 30Hz（高于 15Hz 阈值，确保被判定为"高活动"）
#define HOMEOSTATIC_LR                      0.05f    // 阈值调节学习率（提高 50× 让 IP 在 500 步内生效）
#define HOMEOSTATIC_MAX_OFFSET              2.0f     // 阈值最大偏移量（±2.0）

#define TIME_STEP_MS          1.0f
#define DEFAULT_TIME_STEPS    500

#define N_NEUROMODULATORS     2

#define THREADS_PER_BLOCK     256

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while(0)
#define CUDA_CHECK_LAST() do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while(0)

// =============================================================================
// Stage 2 overrides (active only when SNN_STAGE2_BUILD is defined)
// =============================================================================
// 2c analysis revealed that the default homeostatic settings (target_fr=5Hz,
// LR=0.05, max_offset=2.0) over-stabilize firing: all active neurons converge
// to spike counts within 3 of each other (relative diff 0.09%), erasing any
// byte-neuron selectivity (chi-square: 0/7857 neurons significant).
//
// 2d v1 (P0): relaxed to target_fr=15Hz, LR=0.02, max_offset=0.5.
//   - 100k steps: spike count range expanded to 3333 (success!)
//   - 1M steps: range collapsed back to 3 (homeostatic re-dominated)
//   Diagnosis: threshold_offset accumulates to ±0.5 over long training,
//   which is still strong enough to clamp firing to target_fr.
//
// 2d v2 (P0 re-tune): near-disable homeostatic so STDP fully dominates.
//   - LR=0.005 (4x slower than v1, 10x slower than 2c default)
//   - MAX_OFFSET=0.1 (5x smaller than v1, 20x smaller than 2c default)
//     At max_offset=0.1, threshold only shifts by 0.1/16=0.00625 ~= 0.6%
//     of base threshold -- effectively a residual slow drift corrector,
//     not a fast firing-rate equalizer.
//   - Keep target_fr=15Hz so drift correction targets a reasonable rate.
// =============================================================================
#ifdef SNN_STAGE2_BUILD
  #undef HOMEOSTATIC_TARGET_RATE_SENSORY
  #undef HOMEOSTATIC_TARGET_RATE_ASSOCIATION
  #undef HOMEOSTATIC_LR
  #undef HOMEOSTATIC_MAX_OFFSET
  #define HOMEOSTATIC_TARGET_RATE_SENSORY     0.015f   // 15Hz (was 5Hz)
  #define HOMEOSTATIC_TARGET_RATE_ASSOCIATION 0.015f   // 15Hz (was 5Hz)
  #define HOMEOSTATIC_LR                      0.005f   // v2: was 0.02 (5x slower)
  #define HOMEOSTATIC_MAX_OFFSET              0.1f     // v2: was 0.5 (5x smaller)
#endif

#endif
