// =============================================================================
// Stage 2e 输入编码实现 (P1, §2.3 群体编码)
// =============================================================================
// 设计要点 (Phase R2 模块 C: 注入目标为 L4 层):
//   - input_inject_kernel: 每 thread 处理一个柱, 在柱内 L4 层激活 K 个神经元
//   - L4 层位于柱首 200 神经元 (col_base..col_base+200), 索引计算不变
//   - 哈希: hash(byte, col_seed) 生成 K 个偏移, 写入 input_current[base + offset] += GAIN
//   - 使用 atomicAdd 避免竞争 (K=50, 50 柱并发, 每柱 50 次原子加 = 2500 次/步)
//
// 性能考虑:
//   - 50 柱 × 50 神经元 = 2500 threads (1 block × 256 threads = 11 blocks)
//   - 每步开销小, 不影响主流水线
// =============================================================================

#include "input_encoding.cuh"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// =============================================================================
// UTF-8 文本流加载器 (stage2e 内置, 不依赖 stage2)
// =============================================================================
// 硬约束 (项目记忆): 文本编解码必须使用 8-bit/256 byte (UTF-8 byte stream)
// 设计: 加载 LCCC 子集 (data/lccc_sample_1mb.txt), 保留 UTF-8 多字节序列,
//       仅将 \r\n\t 映射为空格, 丢弃 NUL 字节
// =============================================================================

namespace stage2e {

// 全局文本流缓冲 (host 端, main.cpp 启动时加载一次)
static std::string g_text_buffer;
static size_t g_text_pos = 0;
static bool g_text_loaded = false;

// 加载 UTF-8 文本语料到全局缓冲
// 返回: 加载的字节数 (0 表示失败)
size_t load_text_corpus(const char* filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        std::cerr << "[TextStream] 无法打开文件: " << filepath << std::endl;
        return 0;
    }

    // 读取整个文件为原始字节
    std::string raw((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());

    // 过滤:
    //   - NUL (0x00) -> 丢弃
    //   - \r, \n, \t -> ' ' (0x20)
    //   - 其他字节 (包括 UTF-8 高字节) -> 保留
    g_text_buffer.clear();
    g_text_buffer.reserve(raw.size());
    for (unsigned char c : raw) {
        if (c == 0x00) continue;
        if (c == '\r' || c == '\n' || c == '\t') {
            g_text_buffer.push_back(' ');
        } else {
            g_text_buffer.push_back((char)c);
        }
    }

    g_text_pos = 0;
    g_text_loaded = !g_text_buffer.empty();

    // 统计字节分布 (诊断用)
    int ascii_count = 0, head_count = 0, cont_count = 0;
    for (unsigned char c : g_text_buffer) {
        if (c < 0x80) ascii_count++;
        else if (c >= 0xC0 && c < 0xF8) head_count++;
        else if (c >= 0x80 && c < 0xC0) cont_count++;
    }

    std::cout << "[TextStream] 加载 " << filepath << ": "
              << raw.size() << " 原始字节 -> "
              << g_text_buffer.size() << " 过滤字节" << std::endl;
    std::cout << "  ASCII (<0x80):    " << ascii_count << " ("
              << 100.0 * ascii_count / g_text_buffer.size() << "%)" << std::endl;
    std::cout << "  UTF-8 头字节:      " << head_count << " ("
              << 100.0 * head_count / g_text_buffer.size() << "%)" << std::endl;
    std::cout << "  UTF-8 续字节:      " << cont_count << " ("
              << 100.0 * cont_count / g_text_buffer.size() << "%)" << std::endl;
    std::cout << "  文本预览 (前 80 字节): ";
    for (size_t i = 0; i < 80 && i < g_text_buffer.size(); ++i) {
        unsigned char c = (unsigned char)g_text_buffer[i];
        if (c >= 0x20 && c < 0x7F) std::cout << (char)c;
        else if (c == ' ') std::cout << ' ';
        else std::cout << '?';
    }
    std::cout << std::endl;

    return g_text_buffer.size();
}

// 检查文本是否已加载
bool is_text_loaded() { return g_text_loaded; }

// 获取已加载文本的字节数
size_t text_corpus_size() { return g_text_buffer.size(); }

// 获取文本缓冲指定位置的字节 (越界返回 0)
uint8_t get_text_byte_at(size_t idx) {
    if (idx >= g_text_buffer.size()) return 0;
    return (uint8_t)g_text_buffer[idx];
}

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
    int n_columns,
    int byte_pref_range,    // 每柱偏好字节数 (256 / N_COLUMNS_2E = 5)
    float gain_in,          // 偏好柱增益倍数 (2.0)
    float gain_out,         // 非偏好柱增益倍数 (0.3)
    const ThalamicGateState* gate_states)  // 丘脑门控状态 (每柱一个, nullptr 时全开)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns) return;

    // 柱特异性字节偏好: 柱 c 偏好字节范围 [c*byte_pref_range, (c+1)*byte_pref_range)
    // 偏好柱接收强增益, 非偏好柱接收弱增益, 打破柱间输入对称性
    int pref_start = col * byte_pref_range;
    int pref_end = pref_start + byte_pref_range;
    bool in_range = (byte >= pref_start && byte < pref_end);
    float gain = in_range ? gain_in : gain_out;

    // 丘脑门控调制: gate_signal ∈ [0,1] 缩放增益
    // 向后兼容: nullptr 时全开 (gate=1.0)
    // 注意: ThalamicGateState 是 16B 结构体, 不能简单强转为 float*
    //       必须通过 .gate_signal 字段访问
    float gate = (gate_states != nullptr) ? gate_states[col].gate_signal : 1.0f;

    // 柱内 L4 层基址 (丘脑输入层, 位于柱首 200 神经元, 索引计算不变)
    int col_base = col * NEURONS_PER_COLUMN_2E;  // L4 在柱内最前
    int sensory_base = col_base;  // 0..199 是 L4 (丘脑输入层)

    // 柱特定的哈希种子 (Knuth 乘法常数)
    uint32_t col_seed = static_cast<uint32_t>(col) * 0x9E3779B9u;
    uint32_t hash = static_cast<uint32_t>(byte) * 2654435761u + col_seed;

    // 在柱内 L4 层 (200 神经元) 中激活 K=50 个
    for (int k = 0; k < POP_CODING_K_PER_COLUMN; ++k) {
        // xorshift32 哈希
        hash ^= hash << 13;
        hash ^= hash >> 17;
        hash ^= hash << 5;

        int offset = static_cast<int>(hash % COL_L4_SIZE_2E);
        int neuron_idx = sensory_base + offset;

        // 累积输入电流 (atomicAdd 避免竞争)
        // 丘脑门控: gain * gate 缩放输入 (gate=0 完全闭门, gate=1 全开)
        atomicAdd(&input_current[neuron_idx], POP_CODING_GAIN * gain * gate);
    }
}

// =============================================================================
// Host launcher
// =============================================================================
// d_gate_states: 丘脑门控状态数组 (设备指针), nullptr 时全开 (向后兼容)
void launch_input_inject(MemoryAllocator* alloc, uint8_t byte,
                         const ThalamicGateState* d_gate_states) {
    PersistentBuffers& b = alloc->buffers();

    // 每柱一个 thread, 50 柱 → 1 block (256 threads) 足够
    int threads = (N_COLUMNS_2E <= 256) ? N_COLUMNS_2E : 256;

    input_inject_kernel<<<1, threads>>>(
        b.d_input_current,
        byte,
        N_COLUMNS_2E,
        COLUMN_BYTE_PREF_RANGE,         // 5
        COLUMN_BYTE_PREF_GAIN_IN,       // 2.0f
        COLUMN_BYTE_PREF_GAIN_OUT,      // 0.3f
        d_gate_states);
}

// =============================================================================
// 计算当前步应注入的字节
// =============================================================================
// 真实文本模式: 从 LCCC 语料读取 UTF-8 字节流 (循环)
// 回退模式: 未加载文本时使用 step % 256 循环 (P1 烟雾测试兼容)
uint8_t get_byte_for_step(int step) {
    // 每 INPUT_INJECT_INTERVAL 步注入一个新字节
    if (g_text_loaded && !g_text_buffer.empty()) {
        // 真实文本模式: 从 LCCC 语料循环读取 UTF-8 字节
        uint8_t b = (uint8_t)g_text_buffer[g_text_pos];
        g_text_pos = (g_text_pos + 1) % g_text_buffer.size();
        return b;
    }
    // 回退模式: 0..255 循环 (未加载文本时)
    int byte_idx = (step / INPUT_INJECT_INTERVAL) % INPUT_TEXT_CORPUS_LEN;
    return static_cast<uint8_t>(byte_idx);
}

} // namespace stage2e
