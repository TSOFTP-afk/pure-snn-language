#include "run_config.h"

#include <initializer_list>
#include <iostream>
#include <string>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "CHECK failed at line " << __LINE__ << ": "          \
                      << #condition << '\n';                                    \
            return 1;                                                           \
        }                                                                       \
    } while (false)

static bool parse(std::initializer_list<const char*> args,
                  stage2e::RunConfig* cfg, std::string* error) {
    char* argv[32]{};
    int argc = 0;
    for (const char* arg : args) argv[argc++] = const_cast<char*>(arg);
    return stage2e::parse_run_config(argc, argv, cfg, error);
}

int main() {
    {
        stage2e::RunConfig cfg;
        std::string error;
        CHECK(parse({"test", "--steps", "3000000", "--seed", "7", "--device", "1",
                     "--text", "corpus.txt", "--resume", "checkpoint.bin", "--e0",
                     "--synthetic-input", "--strict-criteria"},
                    &cfg, &error));
        CHECK(cfg.total_steps == 3000000);
        CHECK(cfg.seed == 7);
        CHECK(cfg.device == 1);
        CHECK(cfg.text_path == "corpus.txt");
        CHECK(cfg.resume_path == "checkpoint.bin");
        CHECK(cfg.e0_mode);
        CHECK(cfg.synthetic_input);
        CHECK(cfg.strict_criteria);
    }
    {
        stage2e::RunConfig cfg;
        std::string error;
        CHECK(!parse({"test", "--steps", "0"}, &cfg, &error));
        CHECK(!error.empty());
    }
    {
        stage2e::RunConfig cfg;
        std::string error;
        CHECK(!parse({"test", "--unknown"}, &cfg, &error));
    }
    {
        stage2e::RunConfig cfg;
        std::string error;
        CHECK(parse({"test", "--seed", "4294967295"}, &cfg, &error));
        CHECK(cfg.seed == 4294967295U);
    }
    return 0;
}
