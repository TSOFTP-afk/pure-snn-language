#ifndef SNN_STAGE2E_RUN_CONFIG_H
#define SNN_STAGE2E_RUN_CONFIG_H

#include "config.h"
#include <cstdint>
#include <string>

namespace stage2e {

struct RunConfig {
    int total_steps = 10000;
    int device = 0;
    uint32_t seed = 42;
    uint64_t memory_budget_mb = DEFAULT_VRAM_BUDGET_MB;
    int checkpoint_interval = 50000;
    int keep_checkpoints = 3;
    bool e0_mode = false;
    bool synthetic_input = false;
    bool strict_criteria = false;
    bool show_help = false;
    std::string text_path = "data/lccc_sample_1mb.txt";
    std::string csv_path;
    std::string checkpoint_dir = "checkpoints";
    std::string resume_path;
    // Task 10: 在线解码评估参数
    float decode_lr = 0.001f;           // 解码学习率 (传递给 scheduler, kernel 暂用编译常量)
    bool eval_mode = false;             // 仅推理模式 (不更新 W_decode)
    std::string eval_text_path;         // held-out 评估文本路径 (非空时用于评估)
};

bool parse_run_config(int argc, char** argv, RunConfig* config, std::string* error);
const char* run_config_usage();

} // namespace stage2e

#endif
