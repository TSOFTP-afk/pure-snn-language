// =============================================================================
// main.cpp - stage 2 entry point
// =============================================================================
//
// Sub-stage 2a smoke test + 2b long training launcher.
//
// Default behavior (no args): run STAGE2_SMOKE_TEST_STEPS (= 10000) on the
// built-in fallback corpus, no checkpoint, verbose logging.
//
// Command-line options:
//   --steps N        Number of training steps (default: 10000)
//   --text PATH      Load text corpus from file (default: built-in fallback)
//   --ckpt PATH      Save checkpoint to PATH every STAGE2_CHECKPOINT_INTERVAL
//                    steps and at the end (default: no checkpoint)
//   --resume PATH    Resume from checkpoint at PATH (loads synapses + step
//                    counter + text position, then continues training)
//   --quiet          Suppress per-STAGE2_LOG_INTERVAL logging (only prints
//                    start and end summary)
//   --seed N         Override RNG seed for network init (default: 42)
//   --help           Print usage and exit
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "../include/network.h"
#include "config.h"
#include "columnar_topology.cuh"
#include "text_stream.cuh"
#include "unsupervised_trainer.cuh"

static void print_usage() {
    std::printf(
        "Usage: snn_stage2 [options]\n"
        "\n"
        "Options:\n"
        "  --steps N        Training steps (default: %d)\n"
        "  --text PATH      Load text corpus from file (default: fallback)\n"
        "  --ckpt PATH      Save checkpoint to PATH every %d steps + at end\n"
        "  --resume PATH    Resume from checkpoint at PATH\n"
        "  --quiet          Suppress per-%d-step logging\n"
        "  --seed N         RNG seed (default: 42)\n"
        "  --help           Print this message and exit\n",
        STAGE2_SMOKE_TEST_STEPS,
        STAGE2_CHECKPOINT_INTERVAL,
        STAGE2_LOG_INTERVAL
    );
}

struct CliArgs {
    long        steps        = STAGE2_SMOKE_TEST_STEPS;
    std::string text_path    = "";
    std::string ckpt_path    = "";
    std::string resume_path  = "";
    bool        quiet        = false;
    unsigned int seed        = 42;
    bool        help         = false;
};

static bool parse_args(int argc, char** argv, CliArgs& out) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next_str = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "[Stage2] Missing value for " << name << std::endl;
                return "";
            }
            return std::string(argv[++i]);
        };
        auto next_long = [&](const char* name, long& dst) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "[Stage2] Missing value for " << name << std::endl;
                return false;
            }
            char* end = nullptr;
            long v = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0') {
                std::cerr << "[Stage2] Invalid integer for " << name
                          << ": " << argv[i] << std::endl;
                return false;
            }
            dst = v;
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            out.help = true;
        } else if (arg == "--steps") {
            if (!next_long("--steps", out.steps)) return false;
            if (out.steps < 0) {
                std::cerr << "[Stage2] --steps must be >= 0" << std::endl;
                return false;
            }
        } else if (arg == "--text") {
            out.text_path = next_str("--text");
            if (out.text_path.empty()) return false;
        } else if (arg == "--ckpt") {
            out.ckpt_path = next_str("--ckpt");
            if (out.ckpt_path.empty()) return false;
        } else if (arg == "--resume") {
            out.resume_path = next_str("--resume");
            if (out.resume_path.empty()) return false;
        } else if (arg == "--quiet") {
            out.quiet = true;
        } else if (arg == "--seed") {
            long s = 0;
            if (!next_long("--seed", s)) return false;
            if (s < 0) {
                std::cerr << "[Stage2] --seed must be >= 0" << std::endl;
                return false;
            }
            out.seed = (unsigned int)s;
        } else {
            std::cerr << "[Stage2] Unknown option: " << arg << std::endl;
            std::cerr << "Run with --help for usage." << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }
    if (args.help) {
        print_usage();
        return 0;
    }

    std::printf("============================================================\n");
    std::printf("  SNN Stage 2: Joint Cortex Development\n");
    std::printf("============================================================\n");
    std::printf("  steps    = %ld\n", args.steps);
    std::printf("  text     = %s\n",
                args.text_path.empty() ? "(fallback)" : args.text_path.c_str());
    std::printf("  ckpt     = %s\n",
                args.ckpt_path.empty() ? "(none)" : args.ckpt_path.c_str());
    std::printf("  resume   = %s\n",
                args.resume_path.empty() ? "(none)" : args.resume_path.c_str());
    std::printf("  quiet    = %s\n", args.quiet ? "yes" : "no");
    std::printf("  seed     = %u\n", args.seed);
    std::printf("============================================================\n\n");

    // ---------------------------------------------------------------------
    // 1. Create network + allocate memory (no topology yet)
    // ---------------------------------------------------------------------
    SNNNetwork network(args.seed);
    network.allocate_only();   // stage2 path: skips init_synapses()

    // ---------------------------------------------------------------------
    // 2. Build columnar topology (replaces stage0's init_synapses)
    // ---------------------------------------------------------------------
    init_columnar_synapses(
        network.get_d_synapses(),
        network.get_d_row_ptr(),
        network.get_d_col_idx(),
        network.get_d_weights(),
        N_TOTAL_NEURONS,
        NEURONS_PER_COLUMN,
        INTRA_SYNAPSES_PER_NEURON,
        INTER_SYNAPSES_PER_NEURON,
        SYNAPSES_PER_NEURON,
        /*seed=*/123
    );

    // ---------------------------------------------------------------------
    // 3. Verify topology
    // ---------------------------------------------------------------------
    columnar_topology_report(
        network.get_d_synapses(),
        N_TOTAL_NEURONS,
        NEURONS_PER_COLUMN,
        SYNAPSES_PER_NEURON
    );

    // ---------------------------------------------------------------------
    // 4. Load text stream
    // ---------------------------------------------------------------------
    TextStream stream;
    if (!args.text_path.empty()) {
        if (!stream.load_from_file(args.text_path)) {
            std::cerr << "[Stage2] Falling back to built-in corpus."
                      << std::endl;
            stream.load_fallback();
        }
    } else {
        stream.load_fallback();
    }

    // ---------------------------------------------------------------------
    // 5. Optionally resume from checkpoint
    // ---------------------------------------------------------------------
    long   start_step    = 0;
    size_t saved_text_pos = 0;
    if (!args.resume_path.empty()) {
        if (!load_checkpoint(network, args.resume_path,
                             start_step, saved_text_pos)) {
            std::cerr << "[Stage2] Resume failed; aborting." << std::endl;
            return 1;
        }
        // Advance text stream to saved position
        for (size_t i = 0; i < saved_text_pos; i++) {
            (void)stream.next_byte();
        }
        std::printf("[Stage2] Resumed at step %ld, text_pos %zu\n",
                    start_step, saved_text_pos);
    }

    // ---------------------------------------------------------------------
    // 6. Run training
    // ---------------------------------------------------------------------
    if (args.steps == 0) {
        std::printf("[Stage2] --steps=0, skipping training.\n");
    } else {
        std::printf("\n[Stage2] Starting %ld-step training...\n", args.steps);
        long final_step = run_unsupervised_training(
            network,
            stream,
            start_step,
            args.steps,
            args.ckpt_path,
            /*verbose=*/!args.quiet
        );
        (void)final_step;
    }

    // ---------------------------------------------------------------------
    // 7. Print final summary
    // ---------------------------------------------------------------------
    NetworkStats final_stats;
    network.get_stats(final_stats);

    std::printf("\n============================================================\n");
    std::printf("  Stage 2 training complete\n");
    std::printf("============================================================\n");
    std::printf("  total_spikes      = %d\n", final_stats.total_spikes);
    std::printf("  excitatory_spikes = %d\n", final_stats.excitatory_spikes);
    std::printf("  inhibitory_spikes = %d\n", final_stats.inhibitory_spikes);
    std::printf("  mean_fire_rate    = %.6f\n", final_stats.mean_fire_rate);
    std::printf("  mean_weight       = %.6f\n", final_stats.mean_weight);
    std::printf("============================================================\n");

    network.cleanup();
    return 0;
}
