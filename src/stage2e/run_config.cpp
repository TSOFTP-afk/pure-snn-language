#include "run_config.h"

#include <cerrno>
#include <climits>
#include <cstdlib>

namespace stage2e {
namespace {

bool parse_long(const char* text, long min_value, long max_value,
                const char* option, long* value, std::string* error) {
    if (!text || !*text) {
        *error = std::string(option) + " requires a value";
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < min_value || parsed > max_value) {
        *error = std::string("invalid value for ") + option + ": " + text;
        return false;
    }
    *value = parsed;
    return true;
}

bool parse_u32(const char* text, const char* option, uint32_t* value,
               std::string* error) {
    if (!text || !*text || *text == '-') {
        *error = std::string("invalid value for ") + option;
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed > UINT32_MAX) {
        *error = std::string("invalid value for ") + option + ": " + text;
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

} // namespace

bool parse_run_config(int argc, char** argv, RunConfig* config, std::string* error) {
    if (!config || !error) return false;
    *error = "";

    auto require_value = [&](int* index, const char* option) -> const char* {
        if (*index + 1 >= argc) {
            *error = std::string(option) + " requires a value";
            return nullptr;
        }
        return argv[++(*index)];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const char* value = nullptr;
        long parsed = 0;
        if (arg == "--help" || arg == "-h") {
            config->show_help = true;
        } else if (arg == "--e0") {
            config->e0_mode = true;
        } else if (arg == "--synthetic-input") {
            config->synthetic_input = true;
        } else if (arg == "--steps") {
            value = require_value(&i, "--steps");
            if (!value || !parse_long(value, 1, INT_MAX, "--steps", &parsed, error)) return false;
            config->total_steps = static_cast<int>(parsed);
        } else if (arg == "--device") {
            value = require_value(&i, "--device");
            if (!value || !parse_long(value, 0, INT_MAX, "--device", &parsed, error)) return false;
            config->device = static_cast<int>(parsed);
        } else if (arg == "--seed") {
            value = require_value(&i, "--seed");
            if (!value || !parse_u32(value, "--seed", &config->seed, error)) return false;
        } else if (arg == "--checkpoint-interval") {
            value = require_value(&i, "--checkpoint-interval");
            if (!value || !parse_long(value, 0, INT_MAX, "--checkpoint-interval", &parsed, error)) return false;
            config->checkpoint_interval = static_cast<int>(parsed);
        } else if (arg == "--keep-checkpoints") {
            value = require_value(&i, "--keep-checkpoints");
            if (!value || !parse_long(value, 0, INT_MAX, "--keep-checkpoints", &parsed, error)) return false;
            config->keep_checkpoints = static_cast<int>(parsed);
        } else if (arg == "--text") {
            value = require_value(&i, "--text");
            if (!value) return false;
            config->text_path = value;
        } else if (arg == "--csv") {
            value = require_value(&i, "--csv");
            if (!value) return false;
            config->csv_path = value;
        } else if (arg == "--checkpoint-dir") {
            value = require_value(&i, "--checkpoint-dir");
            if (!value) return false;
            config->checkpoint_dir = value;
        } else if (arg == "--resume") {
            value = require_value(&i, "--resume");
            if (!value) return false;
            config->resume_path = value;
        } else {
            *error = "unknown option: " + arg;
            return false;
        }
    }

    if (config->checkpoint_interval > 0 && config->checkpoint_dir.empty()) {
        *error = "--checkpoint-dir cannot be empty when checkpointing is enabled";
        return false;
    }
    return true;
}

const char* run_config_usage() {
    return
        "Usage: snn_stage2e_p1 [options]\n"
        "  --steps N                 stop at absolute step N (default: 10000)\n"
        "  --device N                CUDA device index (default: 0)\n"
        "  --seed N                  topology seed (default: 42)\n"
        "  --text PATH               UTF-8 byte corpus path\n"
        "  --csv PATH                optional per-step diagnostic CSV\n"
        "  --checkpoint-dir PATH     checkpoint directory\n"
        "  --checkpoint-interval N   save every N steps; 0 disables\n"
        "  --keep-checkpoints N      retain newest N; 0 retains all\n"
        "  --resume PATH             resume a complete Stage 2e checkpoint\n"
        "  --synthetic-input         explicit 0..255 cyclic smoke-test input\n"
        "  --e0                      pure-STDP ablation\n"
        "  -h, --help                show this help\n";
}

} // namespace stage2e
