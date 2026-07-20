// =============================================================================
// Stage 2e 输入编码实现 (P1, §2.3 群体编码)
// =============================================================================
// 设计要点:
//   - input_inject_kernel: 每 thread 处理一个柱, 在柱内 sensory 层激活 K 个神经元
//   - 哈希: hash(byte, col_seed) 生成 K 个偏移, 写入 input_current[base + offset] += GAIN
//   - 使用 atomicAdd 避免竞争 (K=50, 50 柱并发, 每柱 50 次原子加 = 2500 次/步)
//
// 性能考虑:
//   - 50 柱 × 50 神经元 = 2500 threads (1 block × 256 threads = 11 blocks)
//   - 每步开销小, 不影响主流水线
// =============================================================================

#include "input_encoding.cuh"
#include <cstdio>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// input_inject_kernel: 群体编码注入
// =============================================================================
// 每 thread 处理 (col, k) 二维索引, 计算哈希偏移, 累积到 input_current
// 启动配置: grid = (50 柱, K_PER_COLUMN), block = (1, 50)
// 简化: 每 thread 处理一个柱, 内部循环 K 次
// =============================================================================
__global__ void input_inject_kernel(
    float* __restrict__ input_current,
    uint8_t byte,
    int n_columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns) return;

    // 柱内 sensory 层基址
    int col_base = col * NEURONS_PER_COLUMN_2E;  // sensory 在柱内最前
    int sensory_base = col_base;  // 0..199 是 sensory

    // 柱特定的哈希种子 (Knuth 乘法常数)
    uint32_t col_seed = static_cast<uint32_t>(col) * 0x9E3779B9u;
    uint32_t hash = static_cast<uint32_t>(byte) * 2654435761u + col_seed;

    // 在柱内 sensory 层 (200 神经元) 中激活 K=50 个
    for (int k = 0; k < POP_CODING_K_PER_COLUMN; ++k) {
        // xorshift32 哈希
        hash ^= hash << 13;
        hash ^= hash >> 17;
        hash ^= hash << 5;

        int offset = static_cast<int>(hash % COL_SENSORY_SIZE_2E);
        int neuron_idx = sensory_base + offset;

        // 累积输入电流 (atomicAdd 避免竞争)
        atomicAdd(&input_current[neuron_idx], POP_CODING_GAIN);
    }
}

// =============================================================================
// Host launcher
// =============================================================================
void launch_input_inject(MemoryAllocator* alloc, uint8_t byte) {
    PersistentBuffers& b = alloc->buffers();

    int blocks = (N_COLUMNS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    // 每柱一个 thread, 50 柱 → 1 block (256 threads) 足够
    int threads = (N_COLUMNS_2E <= 256) ? N_COLUMNS_2E : 256;

    input_inject_kernel<<<1, threads>>>(
        b.d_input_current,
        byte,
        N_COLUMNS_2E);
}

// =============================================================================
// 计算当前步应注入的字节
// =============================================================================
// P1 阶段: 简单的 step % 256 循环 (伪字节流)
// 后续阶段: 替换为真实文本语料 (DailyDialog / LCCC)
uint8_t get_byte_for_step(int step) {
    // 每 INPUT_INJECT_INTERVAL 步注入一个新字节
    // 字节流: 0, 1, 2, ..., 255, 0, 1, ... (周期 256)
    int byte_idx = (step / INPUT_INJECT_INTERVAL) % INPUT_TEXT_CORPUS_LEN;
    return static_cast<uint8_t>(byte_idx);
}

} // namespace stage2e
