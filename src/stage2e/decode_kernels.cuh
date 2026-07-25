#ifndef SNN_STAGE2E_DECODE_KERNELS_CUH
#define SNN_STAGE2E_DECODE_KERNELS_CUH

// =============================================================================
// Stage 2e 在线解码 kernel 与预测误差驱动学习 (Task 4-5)
// =============================================================================
// 对应设计文档 §4: 在线线性解码器
//
// 数学:
//   前向:  logits[b]   = Σ_i W_decode[i*256+b] · spike_flags[i]
//   概率:  p[b]        = softmax(logits)[b]
//   预测:  pred_byte   = argmax_b p[b]
//   误差:  error[b]    = p[b] - one_hot(b == target_byte)
//   损失:  loss        = -log(p[target_byte] + ε)
//   更新:  ΔW[i,b]     = -η · error[b] · spike_flags[i]
//   归一:  每 100 步对 W_decode 每行 L2 归一化 (||w_i|| ≤ 1)
//
// 缓冲区 (在 memory_allocator.cuh 的 PersistentBuffers 中已分配):
//   - d_decode_weights      [N_TOTAL_NEURONS_2E × 256 float]
//   - d_decode_logits       [256 float]                  (前向后存 softmax 概率)
//   - d_decode_error        [256 float]
//   - d_decode_predicted_byte [1 int]
//   - d_spike_flags         [N_TOTAL_NEURONS_2E bool]
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include <cstdint>

namespace stage2e {

// -----------------------------------------------------------------------------
// CUDA kernel 声明
// -----------------------------------------------------------------------------

// Task 4.1: 前向解码 kernel
//   logits[b] = Σ_i W_decode[i*256+b] · spike_flags[i]
// 启动配置: 1 block, 256 threads (每线程负责一个字节 b)
// 优化: 用 shared memory 分块缓存 spike_flags, 让 256 个线程广播读取
__global__ void decode_forward_kernel(
    const float* __restrict__ decode_weights,  // [N_TOTAL_NEURONS_2E × 256]
    const bool* __restrict__ spike_flags,       // [N_TOTAL_NEURONS_2E]
    float* __restrict__ logits,                 // [256]
    int n_neurons);

// Task 4.2: 数值稳定的 in-place softmax
// 启动配置: 1 block, 256 threads
__global__ void decode_softmax_kernel(float* logits, int n);

// Task 4.3: argmax reduction
// 启动配置: 1 block, 256 threads
__global__ void decode_argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ predicted_byte);

// Task 5.1: 计算预测误差 + cross-entropy loss
// 启动配置: 1 block, 256 threads
__global__ void decode_error_kernel(
    const float* __restrict__ logits,    // [256] softmax 后的概率
    float* __restrict__ error,           // [256] 输出误差
    uint8_t target_byte,                 // 真实输入字节
    float* __restrict__ loss_output);    // [1] 输出 loss

// Task 5.2: 解码权重更新 (ΔW = -η · error · spike_flags)
// 启动配置: grid = ceil(n_neurons / 256), block = 256
__global__ void decode_weight_update_kernel(
    float* __restrict__ decode_weights,  // [N × 256]
    const float* __restrict__ error,     // [256]
    const bool* __restrict__ spike_flags,// [N]
    float learning_rate,
    int n_neurons);

// Task 5.3: 行 L2 归一化 (||w_i||_2 > 1 则归一化到 1)
// 启动配置: grid = n_neurons, block = 256
__global__ void decode_weight_normalize_kernel(
    float* __restrict__ decode_weights,  // [N × 256]
    int n_neurons);

// -----------------------------------------------------------------------------
// Host 端 wrapper 函数
// -----------------------------------------------------------------------------

// Task 4.4: 前向解码链 = forward + softmax + argmax
//   每步都调用, 写入 d_decode_logits (softmax 概率) 与 d_decode_predicted_byte
void launch_decode_forward(PersistentBuffers& buf);

// Task 5.1 host: 计算误差 + 拷贝 loss 到 host
//   out_loss: host 端 float 引用, 用于接收 cross-entropy loss
void launch_decode_error(PersistentBuffers& buf, uint8_t target_byte, float& out_loss);

// Task 5.2 host: 权重更新 (ΔW = -η · error · spike_flags)
void launch_decode_weight_update(PersistentBuffers& buf);

// Task 5.3 host: 行 L2 归一化
void launch_decode_weight_normalize(PersistentBuffers& buf);

} // namespace stage2e

#endif // SNN_STAGE2E_DECODE_KERNELS_CUH
