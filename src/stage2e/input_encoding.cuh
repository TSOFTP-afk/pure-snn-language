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
#include "thalamic_gate.cuh"

namespace stage2e {

// 把一个字节编码为群体编码输入, 累积到 d_input_current (在 delay_inject 之后调用)
// 每 INPUT_INJECT_INTERVAL 步注入一个新字节
// P1 阶段: 用伪字节流 (0..255 循环, 或 step % 256)
//
// 输入: byte (0..255)
// 输出: d_input_current[sensory_idx] += POP_CODING_GAIN * gain * gate
//       (50 柱 × 50 神经元/柱 = 2500 个感觉神经元被激活)
// d_gate_states: 丘脑门控状态数组 (设备指针), nullptr 时全开 (向后兼容)
//   注意: 不能简单强转为 float* 因为 ThalamicGateState 是 16B 结构体,
//   gate_signal 字段在 offset 0 但相邻柱的 gate_signal 间隔 16 字节而非 4 字节
void launch_input_inject(MemoryAllocator* alloc, uint8_t byte,
                         const ThalamicGateState* d_gate_states = nullptr);

// 计算当前步应注入的字节 (P1: 简单的 step % 256 循环)
// 后续阶段: 替换为真实文本语料 (DailyDialog / LCCC)
uint8_t get_byte_for_step(int step);

// 加载 UTF-8 文本语料 (LCCC 子集) 到全局缓冲
// 调用后 get_byte_for_step 将从文本流读取而非 step%256 循环
// 返回: 加载的字节数 (0 表示失败)
size_t load_text_corpus(const char* filepath);

// 检查文本语料是否已加载
bool is_text_loaded();

// 获取已加载文本的字节数
size_t text_corpus_size();

// 获取文本缓冲指定位置的字节 (越界返回 0)
uint8_t get_text_byte_at(size_t idx);

// Checkpoint/resume uses the logical corpus cursor, not a raw host pointer.
size_t text_stream_position();
bool set_text_stream_position(size_t position);
uint64_t text_corpus_fingerprint();

} // namespace stage2e

#endif // SNN_STAGE2E_INPUT_ENCODING_CUH
