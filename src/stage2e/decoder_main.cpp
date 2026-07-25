// =============================================================================
// Stage 2e 离线解码器 - 独立入口
// =============================================================================
// 对应 spec: 修复字节身份区分问题 - 阶段 3 (离线解码器)
//
// 用法:
//   snn_stage2e_decoder --ckpt <checkpoint_path> --text <test_text_path> [--steps N]
//
// 参数:
//   --ckpt <path>   : checkpoint 文件路径 (必需)
//   --text <path>   : 测试文本路径 (必需, 二进制读取)
//   --steps <N>     : 解码步数 (可选, 默认 10000, 即处理前 N 个字节)
//   --threshold <N> : L6 神经元响应阈值 (可选, 默认 5)
//   --help          : 显示帮助
//
// 零侵入保证:
//   - 不修改任何训练流程代码 (scheduler/kernels/main 等)
//   - 仅 read-only 访问 checkpoint 文件
// =============================================================================

#include "decoder.cuh"
#include "config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// -----------------------------------------------------------------------------
// 打印用法
// -----------------------------------------------------------------------------
static void print_usage(const char* prog_name) {
    printf("Stage 2e 离线解码器 (零侵入)\n");
    printf("\n");
    printf("用法: %s --ckpt <checkpoint_path> --text <test_text_path> [选项]\n",
           prog_name);
    printf("\n");
    printf("必需参数:\n");
    printf("  --ckpt <path>    checkpoint 文件路径\n");
    printf("  --text <path>    测试文本路径 (二进制读取)\n");
    printf("\n");
    printf("可选参数:\n");
    printf("  --steps <N>      解码步数 (默认 %d, 处理前 N 个字节)\n",
           10000);
    printf("  --threshold <N>  L6 神经元响应阈值 (默认 5)\n");
    printf("  --help           显示此帮助信息\n");
    printf("\n");
    printf("示例:\n");
    printf("  %s --ckpt checkpoints/ckpt_step10000.bin --text corpus.txt --steps 10000\n",
           prog_name);
}

// -----------------------------------------------------------------------------
// 解析命令行参数
// -----------------------------------------------------------------------------
static bool parse_args(int argc, char** argv, stage2e::DecoderConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return false;  // 触发 usage 显示
        } else if (arg == "--ckpt" && i + 1 < argc) {
            cfg.ckpt_path = argv[++i];
        } else if (arg == "--text" && i + 1 < argc) {
            cfg.text_path = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            cfg.decode_steps = std::atoi(argv[++i]);
            if (cfg.decode_steps < 0) cfg.decode_steps = 10000;
        } else if (arg == "--threshold" && i + 1 < argc) {
            cfg.l6_neuron_threshold = std::atoi(argv[++i]);
            if (cfg.l6_neuron_threshold < 1) cfg.l6_neuron_threshold = 5;
        } else {
            fprintf(stderr, "未知参数: %s\n", arg.c_str());
            return false;
        }
    }

    // 校验必需参数
    if (cfg.ckpt_path.empty()) {
        fprintf(stderr, "错误: 缺少必需参数 --ckpt\n");
        return false;
    }
    if (cfg.text_path.empty()) {
        fprintf(stderr, "错误: 缺少必需参数 --text\n");
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// 主入口
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 解析命令行参数
    stage2e::DecoderConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    printf("============================================================\n");
    printf("Stage 2e 离线解码器\n");
    printf("============================================================\n");
    printf("Checkpoint: %s\n", cfg.ckpt_path.c_str());
    printf("测试文本:   %s\n", cfg.text_path.c_str());
    printf("解码步数:   %d\n", cfg.decode_steps);
    printf("L6 阈值:    %d\n", cfg.l6_neuron_threshold);
    printf("\n");

    // 阶段 1: 从 checkpoint 加载 neuron_byte_counts
    printf("--- 阶段 1: 加载 checkpoint ---\n");
    int n_neurons = 0;
    int ckpt_step = 0;
    std::vector<int> neuron_byte_counts =
        stage2e::load_neuron_byte_counts(cfg.ckpt_path, n_neurons, ckpt_step);
    if (neuron_byte_counts.empty()) {
        fprintf(stderr, "加载 checkpoint 失败, 退出\n");
        return 2;
    }
    printf("成功加载 %d 神经元的字节响应计数 (checkpoint step=%d)\n",
           n_neurons, ckpt_step);
    printf("\n");

    // 阶段 2: 对每个 L6 神经元找其响应最强的字节 (argmax)
    printf("--- 阶段 2: 分析 L6 神经元偏好 ---\n");
    std::vector<int> neuron_best_byte =
        stage2e::find_neuron_best_byte(neuron_byte_counts, n_neurons);
    printf("\n");

    // 阶段 3: 解码测试文本
    printf("--- 阶段 3: 解码测试文本 ---\n");
    stage2e::DecodeResult result =
        stage2e::decode_text_segment(cfg.text_path,
                                      neuron_byte_counts,
                                      neuron_best_byte,
                                      cfg.decode_steps,
                                      cfg.l6_neuron_threshold);
    result.ckpt_step = ckpt_step;
    result.n_neurons = n_neurons;
    printf("\n");

    // 阶段 4: 输出解码报告
    stage2e::print_decode_report(cfg, result);

    // 返回值: 0=成功, 根据准确率判断
    if (result.accuracy > 0.05) {
        return 0;  // 学到初步字节身份
    } else {
        return 0;  // 仍然返回 0 (运行成功), 准确率低不代表程序失败
    }
}
