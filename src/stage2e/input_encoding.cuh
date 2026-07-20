#ifndef SNN_STAGE2E_INPUT_ENCODING_CUH
#define SNN_STAGE2E_INPUT_ENCODING_CUH

// =============================================================================
// Stage 2e 输入编码 (P1, §2.3 群体编码)
// =============================================================================
// 对应设计文档 §2.3: 每个输入字节激活 50 柱 × 50 神经元/柱 = 2500 个感觉神经元
//   - 哈希映射: 群体索引 = hash(byte, column_id) % N_per_column
//   - 柱间差异化: 同一字节在不同柱激活不同神经元子集
//   - 信息密度: 2500 个活跃神经元 / 8 (旧 one-hot) = 312× 提升
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

namespace stage2e {

// 把一个字节编码为群体编码输入, 累积到 d_input_current (在 delay_inject 之后调用)
// 每 INPUT_INJECT_INTERVAL 步注入一个新字节
// P1 阶段: 用伪字节流 (0..255 循环, 或 step % 256)
//
// 输入: byte (0..255)
// 输出: d_input_current[sensory_idx] += POP_CODING_GAIN
//       (50 柱 × 50 神经元/柱 = 2500 个感觉神经元被激活)
void launch_input_inject(MemoryAllocator* alloc, uint8_t byte);

// 计算当前步应注入的字节 (P1: 简单的 step % 256 循环)
// 后续阶段: 替换为真实文本语料 (DailyDialog / LCCC)
uint8_t get_byte_for_step(int step);

} // namespace stage2e

#endif // SNN_STAGE2E_INPUT_ENCODING_CUH
