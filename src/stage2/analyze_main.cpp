// =============================================================================
// analyze_main.cpp - stage 2c structural analysis entry point
// =============================================================================
//
// Usage:
//   snn_stage2_analyze --ckpt <path> --text <path> [--steps N] [--label NAME]
//   snn_stage2_analyze --random --text <path> [--steps N] [--label NAME]
//
// Modes:
//   --ckpt PATH   Load trained network from checkpoint, analyze it.
//   --random      Generate fresh random topology (no training), analyze as B1.
//   (default)     If neither flag given, defaults to --random with label "B1".
//
// Output:
//   - Full analysis report to stdout
//   - CSV file with PCA coordinates + cluster assignments
// =============================================================================

#include "analyzer.cuh"
#include "config.h"
#include "../include/config.h"
#include "../include/network.h"
#include "columnar_topology.cuh"
#include "text_stream.cuh"
#include "unsupervised_trainer.cuh"

#include <iostream>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --ckpt PATH    Load trained network from checkpoint" << std::endl;
    std::cout << "  --random       Use random initial weights (B1 baseline)" << std::endl;
    std::cout << "  --text PATH    Text file for test sequence (required)" << std::endl;
    std::cout << "  --steps N      Test sequence length (default 10000)" << std::endl;
    std::cout << "  --label NAME   Label for this analysis (default: Train or B1)" << std::endl;
    std::cout << "  --csv PATH     Save PCA+cluster CSV to this path" << std::endl;
    std::cout << "  --help         Show this help" << std::endl;
}

int main(int argc, char** argv) {
    std::string ckpt_path;
    std::string text_path;
    std::string label;
    std::string csv_path;
    bool use_random = false;
    long n_steps = 10000;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--ckpt" && i + 1 < argc) {
            ckpt_path = argv[++i];
        } else if (arg == "--random") {
            use_random = true;
        } else if (arg == "--text" && i + 1 < argc) {
            text_path = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            n_steps = std::atol(argv[++i]);
        } else if (arg == "--label" && i + 1 < argc) {
            label = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (text_path.empty()) {
        std::cerr << "Error: --text is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (ckpt_path.empty() && !use_random) {
        std::cerr << "Error: must specify --ckpt PATH or --random" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (label.empty()) {
        label = use_random ? "B1-Random" : "Train";
    }

    // ---- Print config ----
    std::cout << "============================================================" << std::endl;
    std::cout << "  SNN Stage 2c: Structural Analysis" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  label  = " << label << std::endl;
    std::cout << "  ckpt   = " << (ckpt_path.empty() ? "(none, random)" : ckpt_path) << std::endl;
    std::cout << "  text   = " << text_path << std::endl;
    std::cout << "  steps  = " << n_steps << std::endl;
    std::cout << "  csv    = " << (csv_path.empty() ? "(none)" : csv_path) << std::endl;
    std::cout << std::endl;

    // ---- Load text stream ----
    TextStream stream;
    if (!stream.load_from_file(text_path)) {
        std::cerr << "Failed to load text" << std::endl;
        return 1;
    }
    stream.reset();

    // ---- Build network ----
    SNNNetwork network(42);
    network.allocate_only();
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

    // ---- Load checkpoint OR use random weights ----
    if (!ckpt_path.empty()) {
        long step;
        size_t pos;
        if (!load_checkpoint(network, ckpt_path, step, pos)) {
            std::cerr << "Failed to load checkpoint" << std::endl;
            return 1;
        }
        std::cout << "[Analyze] Loaded checkpoint at step " << step << std::endl;
    } else {
        std::cout << "[Analyze] Using random initial weights (B1 baseline)" << std::endl;
    }

    // ---- Run analyses ----
    std::cout << std::endl << "=== Collecting activation data ===" << std::endl;
    ActivationData act = collect_activation_data(network, stream, n_steps);

    std::cout << std::endl << "=== Power-law analysis ===" << std::endl;
    PowerLawResult pl = analyze_power_law(act);

    std::cout << std::endl << "=== Chi-square test ===" << std::endl;
    ChiSquareResult chi = analyze_chi_square(act);

    std::cout << std::endl << "=== PCA + K-means clustering ===" << std::endl;
    ClusterResult clu = analyze_clusters(act);

    std::cout << std::endl << "=== Weight distribution ===" << std::endl;
    WeightStats ws = analyze_weight_distribution(network);

    // ---- Print full report ----
    print_analysis_report(label, act, pl, chi, clu, ws);

    // ---- Save CSV ----
    if (!csv_path.empty()) {
        save_cluster_csv(csv_path, clu);
    }

    // ---- Cleanup ----
    network.cleanup();
    return 0;
}
