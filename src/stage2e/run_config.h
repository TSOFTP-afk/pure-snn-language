#ifndef SNN_STAGE2E_RUN_CONFIG_H
#define SNN_STAGE2E_RUN_CONFIG_H

#include <cstdint>
#include <string>

namespace stage2e {

struct RunConfig {
    int total_steps = 10000;
    int device = 0;
    uint32_t seed = 42;
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
};

bool parse_run_config(int argc, char** argv, RunConfig* config, std::string* error);
const char* run_config_usage();

} // namespace stage2e

#endif
