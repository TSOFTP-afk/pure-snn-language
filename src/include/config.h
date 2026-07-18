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

#endif
