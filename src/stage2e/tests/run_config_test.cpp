#include "run_config.h"

#include <cassert>
#include <initializer_list>
#include <string>

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
        assert(parse({"test", "--steps", "3000000", "--seed", "7", "--device", "1",
                      "--text", "corpus.txt", "--resume", "checkpoint.bin", "--e0",
                      "--synthetic-input"},
                     &cfg, &error));
        assert(cfg.total_steps == 3000000);
        assert(cfg.seed == 7);
        assert(cfg.device == 1);
        assert(cfg.text_path == "corpus.txt");
        assert(cfg.resume_path == "checkpoint.bin");
        assert(cfg.e0_mode);
        assert(cfg.synthetic_input);
    }
    {
        stage2e::RunConfig cfg;
        std::string error;
        assert(!parse({"test", "--steps", "0"}, &cfg, &error));
        assert(!error.empty());
    }
    {
        stage2e::RunConfig cfg;
        std::string error;
        assert(!parse({"test", "--unknown"}, &cfg, &error));
    }
    {
        stage2e::RunConfig cfg;
        std::string error;
        assert(parse({"test", "--seed", "4294967295"}, &cfg, &error));
        assert(cfg.seed == 4294967295U);
    }
    return 0;
}
