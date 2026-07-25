#ifndef SNN_STAGE2E_DECODER_CUH
#define SNN_STAGE2E_DECODER_CUH

// =============================================================================
// Stage 2e 离线解码器 (零侵入)
// =============================================================================
// 对应 spec: 修复字节身份区分问题 - 阶段 3 (离线解码器)
//
// 设计目标:
//   - 零侵入: 不修改任何训练流程代码 (scheduler/kernels/main 等)
//   - 只读访问 checkpoint 文件
//   - 从 checkpoint 读取 d_neuron_byte_counts (55K×256 矩阵)
//   - 对每个 L6 神经元找其响应最强的字节 (argmax)
//   - 对测试文本的每个字节, 用多数投票/阈值判断预测字节
//   - 输出解码准确率、混淆矩阵、Top-10 字节解码效果
//
// 数据来源:
//   checkpoint 文件格式 (v3, 见 scheduler_checkpoint.cu::save_checkpoint):
//     章节化布局, magic="SNN2ECP3", version=3
//     1. CheckpointHeader (56B)
//     2. DiskSection[section_count] (每个 56B, 含 name + bytes)
//     3. payload (各 section 数据连续存储, 含 "neuron_byte_counts" 等)
//     4. CheckpointFooter (16B, magic="SNN2EOK3" + checksum)
//   解码器只读取 "neuron_byte_counts" section:
//     N_TOTAL_NEURONS_2E × 256 × 4B = 56MB
//
// L6 神经元索引范围 (见 network_init.cu):
//   每柱 1000 神经元, 柱内布局: L4(0-199) + L2/3(200-549) + L5(550-749) + L6(750-999)
//   第 col 柱 L6 全局索引: col*1000 + 750 .. col*1000 + 999
//   L6 总数: N_L6_TOTAL_2E = 50 * 250 = 12,500
// =============================================================================

#include "config.h"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace stage2e {

// -----------------------------------------------------------------------------
// 解码器配置
// -----------------------------------------------------------------------------
struct DecoderConfig {
    std::string ckpt_path;          // checkpoint 文件路径 (必需)
    std::string text_path;          // 测试文本路径 (必需)
    int         decode_steps;       // 解码步数 (默认 10000, 即处理前 N 个字节)
    int         l6_neuron_threshold;// L6 神经元响应阈值 (默认 5, 即至少 N 个神经元最佳字节=b 才预测为 b)

    DecoderConfig()
        : decode_steps(10000), l6_neuron_threshold(5) {}
};

// -----------------------------------------------------------------------------
// 解码结果
// -----------------------------------------------------------------------------
struct DecodeResult {
    // 总体准确率
    int    total_bytes;             // 总字节数 (实际处理)
    int    correct_bytes;           // 正确解码字节数
    int    undecodable_bytes;       // 无法解码字节数 (预测为"无法解码")
    double accuracy;                // 准确率 = correct / total
    double random_baseline;         // 随机基线 = 1/256

    // 每字节统计 (索引 0..255)
    std::vector<int> byte_count;        // [256] 每个字节在测试文本中出现次数
    std::vector<int> byte_correct;      // [256] 每个字节正确解码次数
    std::vector<int> byte_undecodable;  // [256] 每个字节无法解码次数

    // 混淆矩阵 (稀疏存储): (真实字节, 预测字节) → 混淆次数
    // 预测字节 256 表示"无法解码"
    std::map<std::pair<int,int>, int> confusion;

    // 训练统计 (来自 checkpoint)
    int    ckpt_step;                // checkpoint 保存时的训练步数
    int    n_neurons;                // 神经元总数 (应为 55,000)
    int    n_l6_neurons;             // L6 神经元总数 (应为 12,500)
    int    n_l6_active;              // 有响应的 L6 神经元数 (row_total > 0)

    // 每个字节的 L6 神经元偏好数 (训练统计)
    // byte_neuron_count[b] = "最佳字节 = b" 的 L6 神经元数量
    std::vector<int> byte_neuron_count;  // [256]

    DecodeResult()
        : total_bytes(0), correct_bytes(0), undecodable_bytes(0),
          accuracy(0.0), random_baseline(1.0/256.0),
          ckpt_step(0), n_neurons(0), n_l6_neurons(0), n_l6_active(0) {
        byte_count.assign(256, 0);
        byte_correct.assign(256, 0);
        byte_undecodable.assign(256, 0);
        byte_neuron_count.assign(256, 0);
    }
};

// -----------------------------------------------------------------------------
// 解码器接口
// -----------------------------------------------------------------------------

// 从 checkpoint 加载 neuron_byte_counts (55K × 256 矩阵)
// 参数:
//   ckpt_path  - checkpoint 文件路径
//   n_neurons  - 输出: 神经元总数 (从头部读取, 应为 55,000)
//   ckpt_step  - 输出: checkpoint 保存时的训练步数
// 返回: host 端 55K×256 的计数矩阵 (行优先, neuron_byte_counts[i*256 + b])
//       失败时返回空向量
std::vector<int> load_neuron_byte_counts(const std::string& ckpt_path,
                                          int& n_neurons,
                                          int& ckpt_step);

// 对每个 L6 神经元找其响应最强的字节 (argmax)
// 参数:
//   neuron_byte_counts - 55K×256 的计数矩阵
//   n_neurons          - 神经元总数
// 返回: 长度 N_L6_TOTAL_2E 的向量, 每个元素是对应 L6 神经元的最佳字节 (0..255)
//       L6 神经元按全局索引顺序排列:
//         第 0 个 = 第 0 柱第 750 号神经元 (全局 750)
//         第 1 个 = 第 0 柱第 751 号神经元 (全局 751)
//         ...
//         第 250 个 = 第 1 柱第 750 号神经元 (全局 1750)
//       若该神经元无响应 (row_total=0), 返回 -1
std::vector<int> find_neuron_best_byte(const std::vector<int>& neuron_byte_counts,
                                        int n_neurons);

// 解码测试文本
// 参数:
//   text_path          - 测试文本路径 (二进制读取)
//   neuron_byte_counts - 55K×256 的计数矩阵 (来自 checkpoint)
//   neuron_best_byte   - 每个 L6 神经元的最佳字节 (来自 find_neuron_best_byte)
//   decode_steps       - 处理前 N 个字节 (0 表示处理全部)
//   l6_threshold       - L6 神经元响应阈值
// 返回: 解码结果
DecodeResult decode_text_segment(const std::string& text_path,
                                  const std::vector<int>& neuron_byte_counts,
                                  const std::vector<int>& neuron_best_byte,
                                  int decode_steps,
                                  int l6_threshold);

// 打印解码报告
void print_decode_report(const DecoderConfig& cfg, const DecodeResult& result);

} // namespace stage2e

#endif // SNN_STAGE2E_DECODER_CUH
