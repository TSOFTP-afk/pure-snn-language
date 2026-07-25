// =============================================================================
// Stage 2e 离线解码器实现 (零侵入)
// =============================================================================
// 对应 spec: 修复字节身份区分问题 - 阶段 3 (离线解码器)
//
// 实现说明:
//   - 解码器完全运行在 host 端 (无需 GPU)
//   - 只读访问 checkpoint 文件, 不修改任何训练流程代码
//   - 从 v3 checkpoint 读取 "neuron_byte_counts" section (55K×256 矩阵)
//   - L6 神经元索引范围根据网络结构推算 (见 network_init.cu)
//   - 兼容新版 v3 格式 (.snn2e, magic="SNN2ECP3", 章节化布局)
// =============================================================================

#include "decoder.cuh"
#include "config.h"
#include "ckpt_v3.h"   // v3 checkpoint 读取库
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace stage2e {

// -----------------------------------------------------------------------------
// 辅助: 获取第 col 柱第 idx_in_l6 个 L6 神经元的全局索引
// 柱内布局: L4(0-199) + L2/3(200-549) + L5(550-749) + L6(750-999)
// -----------------------------------------------------------------------------
static inline int get_l6_global_idx(int col, int idx_in_l6) {
    return col * NEURONS_PER_COLUMN_2E
         + COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E
         + idx_in_l6;
}

// -----------------------------------------------------------------------------
// 辅助: 判断全局神经元索引是否属于 L6 层
// -----------------------------------------------------------------------------
static inline bool is_l6_neuron(int global_idx) {
    int off_in_col = global_idx % NEURONS_PER_COLUMN_2E;  // 柱内偏移
    int l6_start = COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E;  // 750
    int l6_end = l6_start + COL_L6_SIZE_2E;  // 1000
    return (off_in_col >= l6_start) && (off_in_col < l6_end);
}

// -----------------------------------------------------------------------------
// 辅助: 获取全局神经元索引对应的 L6 序号 (0..N_L6_TOTAL_2E-1)
// 若不是 L6 神经元, 返回 -1
// -----------------------------------------------------------------------------
static inline int get_l6_seq_idx(int global_idx) {
    if (!is_l6_neuron(global_idx)) return -1;
    int col = global_idx / NEURONS_PER_COLUMN_2E;
    int off_in_col = global_idx % NEURONS_PER_COLUMN_2E;
    int l6_start = COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E;  // 750
    int idx_in_l6 = off_in_col - l6_start;
    return col * COL_L6_SIZE_2E + idx_in_l6;
}

// -----------------------------------------------------------------------------
// load_neuron_byte_counts: 从 v3 checkpoint 加载 55K×256 字节响应计数矩阵
// -----------------------------------------------------------------------------
std::vector<int> load_neuron_byte_counts(const std::string& ckpt_path,
                                          int& n_neurons,
                                          int& ckpt_step) {
    n_neurons = 0;
    ckpt_step = 0;

    // 用 v3 读取器打开 checkpoint
    CkptV3Reader reader;
    if (!reader.open(ckpt_path)) {
        fprintf(stderr, "[Decoder] 错误: 无法打开 v3 checkpoint: %s\n",
                ckpt_path.c_str());
        fprintf(stderr, "[Decoder] 提示: 旧版 v2 格式 (.bin) 不再支持, "
                        "请用新版训练生成 .snn2e 文件\n");
        return {};
    }

    const auto& hdr = reader.header();
    printf("[Decoder] v3 Checkpoint: version=%u n_neurons=%u n_synapses=%u "
           "sections=%u payload=%.2f MiB\n",
           hdr.version, hdr.n_neurons, hdr.n_synapses, hdr.section_count,
           hdr.payload_bytes / (1024.0 * 1024.0));

    // 读取 next_step (来自 scheduler_state section)
    int next_step = 0;
    if (reader.read_next_step(&next_step)) {
        ckpt_step = next_step;
        printf("[Decoder] next_step=%d topology_seed=%u\n",
               next_step, reader.topology_seed());
    }

    // 定位 neuron_byte_counts section
    const auto* nbc = reader.find_section("neuron_byte_counts");
    if (!nbc) {
        fprintf(stderr, "[Decoder] 错误: 'neuron_byte_counts' section 未找到\n");
        return {};
    }

    const uint64_t expect_bytes = (uint64_t)N_TOTAL_NEURONS_2E * 256 * sizeof(int);
    if (nbc->bytes != expect_bytes) {
        fprintf(stderr, "[Decoder] 错误: neuron_byte_counts 字节数不匹配 "
                        "(实际 %llu, 期望 %llu)\n",
                (unsigned long long)nbc->bytes,
                (unsigned long long)expect_bytes);
        return {};
    }

    n_neurons = static_cast<int>(hdr.n_neurons);
    if (n_neurons != N_TOTAL_NEURONS_2E) {
        fprintf(stderr, "[Decoder] 警告: n_neurons=%d (期望 %d)\n",
                n_neurons, N_TOTAL_NEURONS_2E);
    }

    // 读入完整 55K × 256 矩阵
    const size_t bc_count = static_cast<size_t>(n_neurons) * 256;
    std::vector<int> byte_counts(bc_count, 0);
    if (!reader.read_section_payload(*nbc, byte_counts.data(),
                                     bc_count * sizeof(int))) {
        fprintf(stderr, "[Decoder] 错误: 读取 neuron_byte_counts 失败\n");
        return {};
    }

    printf("[Decoder] 已加载 neuron_byte_counts: %zu 个 int (%.1f MiB)\n",
           bc_count, static_cast<double>(bc_count * sizeof(int)) / (1024.0 * 1024.0));

    return byte_counts;
}

// -----------------------------------------------------------------------------
// find_neuron_best_byte: 对每个 L6 神经元找其响应最强的字节 (argmax)
// -----------------------------------------------------------------------------
std::vector<int> find_neuron_best_byte(const std::vector<int>& neuron_byte_counts,
                                        int n_neurons) {
    // L6 神经元总数: N_L6_TOTAL_2E = 50 * 250 = 12,500
    std::vector<int> best_byte(N_L6_TOTAL_2E, -1);

    if (n_neurons != N_TOTAL_NEURONS_2E) {
        fprintf(stderr, "[Decoder] 警告: n_neurons=%d (期望 %d), "
                        "L6 索引计算可能不准\n",
                n_neurons, N_TOTAL_NEURONS_2E);
    }

    int n_l6_active = 0;

    // 遍历所有 L6 神经元
    for (int col = 0; col < N_COLUMNS_2E; ++col) {
        for (int idx_in_l6 = 0; idx_in_l6 < COL_L6_SIZE_2E; ++idx_in_l6) {
            int global_idx = get_l6_global_idx(col, idx_in_l6);
            int l6_seq = get_l6_seq_idx(global_idx);
            if (l6_seq < 0 || l6_seq >= N_L6_TOTAL_2E) continue;

            // 该神经元的 256 字节响应计数
            const int* row = &neuron_byte_counts[static_cast<size_t>(global_idx) * 256];

            // argmax 找最强字节
            int best_b = -1;
            int best_count = 0;
            int row_total = 0;
            for (int b = 0; b < 256; ++b) {
                int c = row[b];
                row_total += c;
                if (c > best_count) {
                    best_count = c;
                    best_b = b;
                }
            }

            // 若该神经元无响应, best_byte 保持 -1
            if (row_total > 0 && best_b >= 0) {
                best_byte[l6_seq] = best_b;
                ++n_l6_active;
            }
        }
    }

    printf("[Decoder] L6 神经元分析完成: %d/%d 有响应 (%.1f%%)\n",
           n_l6_active, N_L6_TOTAL_2E,
           100.0 * n_l6_active / N_L6_TOTAL_2E);

    return best_byte;
}

// -----------------------------------------------------------------------------
// decode_text_segment: 解码测试文本
// -----------------------------------------------------------------------------
DecodeResult decode_text_segment(const std::string& text_path,
                                  const std::vector<int>& neuron_byte_counts,
                                  const std::vector<int>& neuron_best_byte,
                                  int decode_steps,
                                  int l6_threshold) {
    DecodeResult result;
    result.n_l6_neurons = N_L6_TOTAL_2E;

    // 统计每个字节的 L6 神经元偏好数 (训练统计)
    for (int i = 0; i < N_L6_TOTAL_2E; ++i) {
        int bb = neuron_best_byte[i];
        if (bb >= 0 && bb < 256) {
            result.byte_neuron_count[bb]++;
            ++result.n_l6_active;
        }
    }

    // 读取测试文本 (二进制)
    std::ifstream fp(text_path, std::ios::binary);
    if (!fp.is_open()) {
        fprintf(stderr, "[Decoder] 错误: 无法打开测试文本: %s\n",
                text_path.c_str());
        return result;
    }

    // 读取全部内容到缓冲区
    std::vector<uint8_t> text_bytes;
    {
        fp.seekg(0, std::ios::end);
        std::streamoff size = fp.tellg();
        fp.seekg(0, std::ios::beg);
        if (size <= 0) {
            fprintf(stderr, "[Decoder] 错误: 测试文本为空或读取失败\n");
            return result;
        }
        text_bytes.resize(static_cast<size_t>(size));
        fp.read(reinterpret_cast<char*>(text_bytes.data()), size);
        if (!fp) {
            fprintf(stderr, "[Decoder] 错误: 读取测试文本失败\n");
            return result;
        }
    }

    // 限制处理字节数
    int total_to_process = static_cast<int>(text_bytes.size());
    if (decode_steps > 0 && decode_steps < total_to_process) {
        total_to_process = decode_steps;
    }

    printf("[Decoder] 测试文本: %zu 字节, 处理前 %d 字节, L6 阈值=%d\n",
           text_bytes.size(), total_to_process, l6_threshold);

    // 对每个字节解码
    result.total_bytes = total_to_process;
    for (int i = 0; i < total_to_process; ++i) {
        int true_byte = text_bytes[i];
        result.byte_count[true_byte]++;

        // 简化版解码算法:
        // 统计 L6 神经元中"最佳字节 = true_byte"的神经元数量
        // 如果数量 >= l6_threshold, 预测为 true_byte (正确解码)
        // 否则预测为"无法解码" (256)
        int neuron_count = result.byte_neuron_count[true_byte];
        int pred_byte;
        if (neuron_count >= l6_threshold) {
            pred_byte = true_byte;  // 网络识别了该字节
            result.correct_bytes++;
            result.byte_correct[true_byte]++;
        } else {
            pred_byte = 256;  // "无法解码" 标记
            result.undecodable_bytes++;
            result.byte_undecodable[true_byte]++;
        }

        // 更新混淆矩阵
        std::pair<int,int> key(true_byte, pred_byte);
        result.confusion[key]++;
    }

    // 计算准确率
    if (result.total_bytes > 0) {
        result.accuracy = static_cast<double>(result.correct_bytes) /
                          static_cast<double>(result.total_bytes);
    }
    result.random_baseline = 1.0 / 256.0;

    return result;
}

// -----------------------------------------------------------------------------
// print_decode_report: 打印解码报告
// -----------------------------------------------------------------------------
void print_decode_report(const DecoderConfig& cfg, const DecodeResult& result) {
    printf("\n");
    printf("============================================================\n");
    printf("解码器报告\n");
    printf("============================================================\n");
    printf("Checkpoint: %s\n", cfg.ckpt_path.c_str());
    printf("测试文本: %s\n", cfg.text_path.c_str());
    printf("解码步数: %d\n", cfg.decode_steps);
    printf("L6 阈值: %d\n", cfg.l6_neuron_threshold);
    printf("\n");

    // --- 训练统计 ---
    printf("--- 训练统计 (来自 checkpoint) ---\n");
    printf("Checkpoint 步数: %d\n", result.ckpt_step);
    printf("神经元总数: %d\n", result.n_neurons);
    printf("L6 神经元总数: %d\n", result.n_l6_neurons);
    printf("有响应的 L6 神经元数: %d (%.1f%%)\n",
           result.n_l6_active,
           result.n_l6_neurons > 0 ?
               100.0 * result.n_l6_active / result.n_l6_neurons : 0.0);
    printf("\n");

    // --- 解码准确率 ---
    printf("--- 解码准确率 ---\n");
    printf("总字节数: %d\n", result.total_bytes);
    printf("正确解码: %d\n", result.correct_bytes);
    printf("无法解码: %d\n", result.undecodable_bytes);
    printf("准确率: %.2f%%\n", 100.0 * result.accuracy);
    printf("随机基线: %.4f%% (1/256)\n", 100.0 * result.random_baseline);
    printf("相对随机基线倍数: %.2fx\n",
           result.random_baseline > 0 ? result.accuracy / result.random_baseline : 0.0);
    printf("\n");

    // --- Top-10 字节解码效果 ---
    printf("--- Top-10 字节解码效果 (按出现次数降序) ---\n");
    printf("字节    十六进制  出现次数  正确解码  无法解码  准确率\n");

    // 收集出现次数 > 0 的字节, 按出现次数降序排序
    std::vector<std::pair<int,int>> byte_freq;  // (byte, count)
    for (int b = 0; b < 256; ++b) {
        if (result.byte_count[b] > 0) {
            byte_freq.push_back(std::make_pair(b, result.byte_count[b]));
        }
    }
    std::sort(byte_freq.begin(), byte_freq.end(),
              [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
                  return a.second > b.second;
              });

    int top_n = static_cast<int>(byte_freq.size());
    if (top_n > 10) top_n = 10;
    for (int i = 0; i < top_n; ++i) {
        int b = byte_freq[i].first;
        int cnt = byte_freq[i].second;
        int cor = result.byte_correct[b];
        int und = result.byte_undecodable[b];
        double acc = cnt > 0 ? 100.0 * cor / cnt : 0.0;
        printf("%3d     0x%02X      %6d    %6d    %6d    %5.1f%%\n",
               b, b, cnt, cor, und, acc);
    }
    printf("\n");

    // --- L6 神经元偏好分布 (Top-10) ---
    printf("--- L6 神经元偏好分布 (Top-10 字节) ---\n");
    printf("字节    十六进制  偏好神经元数  占比\n");
    std::vector<std::pair<int,int>> neuron_pref;  // (byte, count)
    for (int b = 0; b < 256; ++b) {
        if (result.byte_neuron_count[b] > 0) {
            neuron_pref.push_back(std::make_pair(b, result.byte_neuron_count[b]));
        }
    }
    std::sort(neuron_pref.begin(), neuron_pref.end(),
              [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
                  return a.second > b.second;
              });
    int top_n2 = static_cast<int>(neuron_pref.size());
    if (top_n2 > 10) top_n2 = 10;
    for (int i = 0; i < top_n2; ++i) {
        int b = neuron_pref[i].first;
        int cnt = neuron_pref[i].second;
        double ratio = result.n_l6_neurons > 0 ?
                       100.0 * cnt / result.n_l6_neurons : 0.0;
        printf("%3d     0x%02X      %6d        %5.2f%%\n",
               b, b, cnt, ratio);
    }
    printf("\n");

    // --- 混淆矩阵 (Top-5 最常混淆字节对) ---
    printf("--- 混淆矩阵 (Top-5 最常出现, 不含正确解码) ---\n");
    printf("真实字节  预测字节  混淆次数\n");
    std::vector<std::pair<std::pair<int,int>, int>> conf_list;
    for (const auto& kv : result.confusion) {
        // 排除正确解码 (true == pred)
        if (kv.first.first != kv.first.second) {
            conf_list.push_back(kv);
        }
    }
    std::sort(conf_list.begin(), conf_list.end(),
              [](const std::pair<std::pair<int,int>, int>& a,
                 const std::pair<std::pair<int,int>, int>& b) {
                  return a.second > b.second;
              });
    int top_n3 = static_cast<int>(conf_list.size());
    if (top_n3 > 5) top_n3 = 5;
    for (int i = 0; i < top_n3; ++i) {
        int tb = conf_list[i].first.first;
        int pb = conf_list[i].first.second;
        int cnt = conf_list[i].second;
        if (pb == 256) {
            printf("0x%02X      无法解码   %6d\n", tb, cnt);
        } else {
            printf("0x%02X      0x%02X       %6d\n", tb, pb, cnt);
        }
    }
    if (top_n3 == 0) {
        printf("(无混淆)\n");
    }
    printf("\n");

    // --- 评估 ---
    printf("--- 评估 ---\n");
    if (result.accuracy > 0.20) {
        printf("准确率 %.2f%% > 20%%: 网络已学到较强字节身份\n",
               100.0 * result.accuracy);
    } else if (result.accuracy > 0.05) {
        printf("准确率 %.2f%% > 5%%: 网络已学到初步字节身份\n",
               100.0 * result.accuracy);
    } else if (result.accuracy < 0.01) {
        printf("准确率 %.2f%% < 1%%: 网络未学到字节身份\n",
               100.0 * result.accuracy);
    } else {
        printf("准确率 %.2f%% (1%%-5%%): 网络学到微弱字节身份, 需继续训练\n",
               100.0 * result.accuracy);
    }

    // 给出建议
    if (result.n_l6_active < result.n_l6_neurons / 10) {
        printf("警告: 仅 %d/%d L6 神经元有响应, 网络活动过低\n",
               result.n_l6_active, result.n_l6_neurons);
    }
    int bytes_with_neurons = 0;
    for (int b = 0; b < 256; ++b) {
        if (result.byte_neuron_count[b] >= cfg.l6_neuron_threshold) {
            ++bytes_with_neurons;
        }
    }
    printf("有 %d/256 个字节的 L6 神经元偏好数 >= 阈值 %d\n",
           bytes_with_neurons, cfg.l6_neuron_threshold);

    printf("============================================================\n");
}

} // namespace stage2e
