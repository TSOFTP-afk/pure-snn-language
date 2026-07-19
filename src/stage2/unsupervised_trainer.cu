// =============================================================================
// unsupervised_trainer.cu - stage 2 training loop + checkpoint I/O
// =============================================================================
//
// Pure unsupervised STDP training. No labels, no reward (dopamine stays at
// 1.0 = no modulation). The network learns purely from spike-timing statistics
// of the input character stream.
//
// Per project memory: stage2 must not modify stage0's trainer.cpp. All new
// training logic lives here. Checkpoint saves the full d_synapses_ array
// (32 MB, including STDP state: last_pre_spike, last_post_spike, eligibility)
// so long training runs can resume cleanly.
// =============================================================================

#include "unsupervised_trainer.cuh"
#include "config.h"
#include "../include/config.h"        // N_SENSORY_NEURONS, N_TOTAL_SYNAPSES, ...
#include "../include/types.h"
#include "../include/io.cuh"          // compute_stats()
#include "../include/synapse.cuh"     // sync_weights()
#include "competition.cuh"            // P2: k-WTA columnar competition

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Helper: print one-line training log
// -----------------------------------------------------------------------------
// current_byte may be any value 0x00..0xFF (UTF-8 byte). We display:
//   - printable ASCII (0x20..0x7E) as the char itself
//   - high bytes (0x80..0xFF, UTF-8 multibyte head/continuation) as hex \xNN
//   - control bytes (0x01..0x1F) as hex \xNN
//   - space (0x20) as '.'
static void print_log(long step, const NetworkStats& stats,
                      unsigned char current_byte, long sensory_spikes,
                      long association_spikes, long motor_spikes) {
    char byte_str[8];
    if (current_byte == 0x20) {
        std::snprintf(byte_str, sizeof(byte_str), ".");
    } else if (current_byte >= 0x20 && current_byte <= 0x7E) {
        std::snprintf(byte_str, sizeof(byte_str), "%c", current_byte);
    } else {
        std::snprintf(byte_str, sizeof(byte_str), "\\x%02X", current_byte);
    }

    std::cout << "[step " << step
              << "] spikes(total=" << stats.total_spikes
              << " exc=" << stats.excitatory_spikes
              << " inh=" << stats.inhibitory_spikes
              << ") S=" << sensory_spikes
              << " A=" << association_spikes
              << " M=" << motor_spikes
              << " meanFR=" << stats.mean_fire_rate
              << " meanW=" << stats.mean_weight
              << " byte=" << byte_str
              << std::endl;
}

// -----------------------------------------------------------------------------
// Helper: count spikes per brain region by scanning d_spikes_ on host
// -----------------------------------------------------------------------------
static void count_region_spikes(const bool* d_spikes,
                                long& out_sensory,
                                long& out_association,
                                long& out_motor) {
    bool* h_spikes = new bool[N_TOTAL_NEURONS];
    CUDA_CHECK(cudaMemcpy(h_spikes, d_spikes,
                          N_TOTAL_NEURONS * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    out_sensory = 0;
    out_association = 0;
    out_motor = 0;
    for (int i = 0; i < N_SENSORY_NEURONS; i++) {
        if (h_spikes[i]) out_sensory++;
    }
    int assoc_start = N_SENSORY_NEURONS;
    int assoc_end   = N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS;
    for (int i = assoc_start; i < assoc_end; i++) {
        if (h_spikes[i]) out_association++;
    }
    int motor_start = assoc_end;
    for (int i = motor_start; i < N_TOTAL_NEURONS; i++) {
        if (h_spikes[i]) out_motor++;
    }
    delete[] h_spikes;
}

// =============================================================================
// Main training loop
// =============================================================================
long run_unsupervised_training(
    SNNNetwork& network,
    TextStream& stream,
    long  start_step,
    long  n_steps,
    const std::string& ckpt_path,
    bool  verbose
) {
    std::cout << "[Stage2-Trainer] Starting unsupervised training:" << std::endl;
    std::cout << "  start_step = " << start_step << std::endl;
    std::cout << "  n_steps    = " << n_steps << std::endl;
    std::cout << "  ckpt_path  = "
              << (ckpt_path.empty() ? "(none)" : ckpt_path) << std::endl;
    std::cout << "  text_size  = " << stream.size() << " chars" << std::endl;

    // Sensory input buffer (host)
    // 方案 A: 输入缓冲区长度 = N_TOTAL_NEURONS（sensory 神经元分散在每柱内）
    float* h_input = new float[N_TOTAL_NEURONS];

    long end_step = start_step + n_steps;
    NetworkStats stats;

    for (long t = start_step; t < end_step; t++) {
        // 1. Pull next byte & build sensory input (8-bit UTF-8 byte stream)
        unsigned char b = stream.next_byte();
        stream.build_sensory_input(b, h_input);

        // 2. Single network step (stage0's step() handles STDP + sync)
        network.step(h_input, (int)t);

        // 纯 SNN 实验：移除 k-WTA 外部竞争。
        // 原先: apply_kwta_competition(network, STAGE2_KWTA_K);
        // 让网络内部动力学自己决定柱间竞争（通过抑制性突触）。
        // 如果网络能自发形成功能柱分化，那是涌现；学不到则反映 SNN 真实能力。

        // 3. Periodic logging
        if (verbose && ((t - start_step) % STAGE2_LOG_INTERVAL == 0
                        || t == start_step)) {
            network.get_stats(stats);
            long s_spikes, a_spikes, m_spikes;
            count_region_spikes(network.get_d_spikes(),
                                s_spikes, a_spikes, m_spikes);
            print_log(t, stats, b, s_spikes, a_spikes, m_spikes);
        }

        // 4. Periodic checkpoint
        if (!ckpt_path.empty()
            && (t - start_step) > 0
            && (t - start_step) % STAGE2_CHECKPOINT_INTERVAL == 0) {
            if (!save_checkpoint(network, ckpt_path, t + 1, stream.position())) {
                std::cerr << "[Stage2-Trainer] Checkpoint save FAILED at step "
                          << t << std::endl;
            }
        }

        // 5. Periodic network reset (preserves synapses / weights, clears
        //    membrane potentials, currents, spikes). Prevents runaway
        //    activity and STDP saturation over long runs.
        if ((t - start_step) > 0
            && (t - start_step) % STAGE2_RESET_INTERVAL == 0) {
            network.reset();
        }
    }

    // Final checkpoint
    if (!ckpt_path.empty()) {
        if (!save_checkpoint(network, ckpt_path, end_step, stream.position())) {
            std::cerr << "[Stage2-Trainer] Final checkpoint save FAILED"
                      << std::endl;
        }
    }

    delete[] h_input;
    std::cout << "[Stage2-Trainer] Done. end_step = " << end_step << std::endl;
    return end_step;
}

// =============================================================================
// Checkpoint I/O
// =============================================================================
// Format:
//   [magic     : uint32]   = STAGE2_CHECKPOINT_MAGIC
//   [version   : uint32]   = STAGE2_CHECKPOINT_VERSION
//   [step      : int64]
//   [text_pos  : uint64]
//   [n_neurons : int32]
//   [n_synapses: int32]
//   [d_synapses: Synapse[n_synapses]]   (32B each = 32 MB total)
//   [d_row_ptr : int[n_neurons + 1]]
//   [d_col_idx : int[n_synapses]]
//   [d_weights : float[n_synapses]]
// =============================================================================

bool save_checkpoint(
    SNNNetwork& network,
    const std::string& path,
    long step,
    size_t text_pos
) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        std::cerr << "[Stage2-Ckpt] Cannot open " << path << " for write"
                  << std::endl;
        return false;
    }

    uint32_t magic   = STAGE2_CHECKPOINT_MAGIC;
    uint32_t version = STAGE2_CHECKPOINT_VERSION;
    int64_t  step64  = (int64_t)step;
    uint64_t pos64   = (uint64_t)text_pos;
    int32_t  n_neu   = (int32_t)N_TOTAL_NEURONS;
    int32_t  n_syn   = (int32_t)N_TOTAL_SYNAPSES;

    ofs.write(reinterpret_cast<const char*>(&magic),   sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&step64),  sizeof(step64));
    ofs.write(reinterpret_cast<const char*>(&pos64),   sizeof(pos64));
    ofs.write(reinterpret_cast<const char*>(&n_neu),   sizeof(n_neu));
    ofs.write(reinterpret_cast<const char*>(&n_syn),   sizeof(n_syn));

    // Pull device arrays into host memory and write them.
    // Per project memory: MUST save d_synapses_ (32 MB, incl STDP state),
    // not just d_weights_ (4 MB). stage0's save_weights() only saved
    // d_weights_ and was insufficient for resuming long training runs.

    // d_synapses_
    Synapse* h_syn = new Synapse[N_TOTAL_SYNAPSES];
    CUDA_CHECK(cudaMemcpy(h_syn, network.get_d_synapses(),
                          N_TOTAL_SYNAPSES * sizeof(Synapse),
                          cudaMemcpyDeviceToHost));
    ofs.write(reinterpret_cast<const char*>(h_syn),
              N_TOTAL_SYNAPSES * sizeof(Synapse));
    delete[] h_syn;

    // d_row_ptr_
    int* h_row = new int[N_TOTAL_NEURONS + 1];
    CUDA_CHECK(cudaMemcpy(h_row, network.get_d_row_ptr(),
                          (N_TOTAL_NEURONS + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    ofs.write(reinterpret_cast<const char*>(h_row),
              (N_TOTAL_NEURONS + 1) * sizeof(int));
    delete[] h_row;

    // d_col_idx_
    int* h_col = new int[N_TOTAL_SYNAPSES];
    CUDA_CHECK(cudaMemcpy(h_col, network.get_d_col_idx(),
                          N_TOTAL_SYNAPSES * sizeof(int),
                          cudaMemcpyDeviceToHost));
    ofs.write(reinterpret_cast<const char*>(h_col),
              N_TOTAL_SYNAPSES * sizeof(int));
    delete[] h_col;

    // d_weights_
    float* h_w = new float[N_TOTAL_SYNAPSES];
    CUDA_CHECK(cudaMemcpy(h_w, network.get_d_weights(),
                          N_TOTAL_SYNAPSES * sizeof(float),
                          cudaMemcpyDeviceToHost));
    ofs.write(reinterpret_cast<const char*>(h_w),
              N_TOTAL_SYNAPSES * sizeof(float));
    delete[] h_w;

    std::cout << "[Stage2-Ckpt] Saved step=" << step
              << " text_pos=" << text_pos
              << " to " << path
              << " (~32 MB)" << std::endl;
    return true;
}

bool load_checkpoint(
    SNNNetwork& network,
    const std::string& path,
    long& out_step,
    size_t& out_text_pos
) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[Stage2-Ckpt] Cannot open " << path << " for read"
                  << std::endl;
        return false;
    }

    uint32_t magic, version;
    int64_t  step64;
    uint64_t pos64;
    int32_t  n_neu, n_syn;

    ifs.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&step64),  sizeof(step64));
    ifs.read(reinterpret_cast<char*>(&pos64),   sizeof(pos64));
    ifs.read(reinterpret_cast<char*>(&n_neu),   sizeof(n_neu));
    ifs.read(reinterpret_cast<char*>(&n_syn),   sizeof(n_syn));

    if (!ifs || magic != STAGE2_CHECKPOINT_MAGIC) {
        std::cerr << "[Stage2-Ckpt] Bad magic in " << path << std::endl;
        return false;
    }
    if (version != STAGE2_CHECKPOINT_VERSION) {
        std::cerr << "[Stage2-Ckpt] Version mismatch: file=" << version
                  << " code=" << STAGE2_CHECKPOINT_VERSION << std::endl;
        return false;
    }
    if (n_neu != N_TOTAL_NEURONS || n_syn != N_TOTAL_SYNAPSES) {
        std::cerr << "[Stage2-Ckpt] Size mismatch: file n_neu=" << n_neu
                  << " n_syn=" << n_syn
                  << " code n_neu=" << N_TOTAL_NEURONS
                  << " n_syn=" << N_TOTAL_SYNAPSES << std::endl;
        return false;
    }

    // d_synapses_
    Synapse* h_syn = new Synapse[N_TOTAL_SYNAPSES];
    ifs.read(reinterpret_cast<char*>(h_syn),
             N_TOTAL_SYNAPSES * sizeof(Synapse));
    CUDA_CHECK(cudaMemcpy(network.get_d_synapses(), h_syn,
                          N_TOTAL_SYNAPSES * sizeof(Synapse),
                          cudaMemcpyHostToDevice));
    delete[] h_syn;

    // d_row_ptr_
    int* h_row = new int[N_TOTAL_NEURONS + 1];
    ifs.read(reinterpret_cast<char*>(h_row),
             (N_TOTAL_NEURONS + 1) * sizeof(int));
    CUDA_CHECK(cudaMemcpy(network.get_d_row_ptr(), h_row,
                          (N_TOTAL_NEURONS + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    delete[] h_row;

    // d_col_idx_
    int* h_col = new int[N_TOTAL_SYNAPSES];
    ifs.read(reinterpret_cast<char*>(h_col),
             N_TOTAL_SYNAPSES * sizeof(int));
    CUDA_CHECK(cudaMemcpy(network.get_d_col_idx(), h_col,
                          N_TOTAL_SYNAPSES * sizeof(int),
                          cudaMemcpyHostToDevice));
    delete[] h_col;

    // d_weights_
    float* h_w = new float[N_TOTAL_SYNAPSES];
    ifs.read(reinterpret_cast<char*>(h_w),
             N_TOTAL_SYNAPSES * sizeof(float));
    CUDA_CHECK(cudaMemcpy(network.get_d_weights(), h_w,
                          N_TOTAL_SYNAPSES * sizeof(float),
                          cudaMemcpyHostToDevice));
    delete[] h_w;

    out_step     = (long)step64;
    out_text_pos = (size_t)pos64;
    std::cout << "[Stage2-Ckpt] Loaded step=" << out_step
              << " text_pos=" << out_text_pos
              << " from " << path << std::endl;
    return true;
}
