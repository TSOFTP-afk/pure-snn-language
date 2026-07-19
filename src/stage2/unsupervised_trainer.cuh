#ifndef SNN_STAGE2_UNSUPERVISED_TRAINER_CUH
#define SNN_STAGE2_UNSUPERVISED_TRAINER_CUH

#include <string>
#include "../include/network.h"
#include "text_stream.cuh"

// =============================================================================
// Stage 2 unsupervised trainer
// =============================================================================
// Replaces stage0's trainer.cpp (which was supervised + reward-based) with a
// pure unsupervised STDP loop:
//   1. Pull next char from TextStream
//   2. Build 7-bit sensory input vector (rest of sensory cortex zeroed)
//   3. Call SNNNetwork::step(input, t) -- stage0's STDP kernel learns
//      based on spike timing, no labels, no reward
//   4. Periodically log stats, save checkpoint, reset network state
//
// Hard constraints (project memory):
//   - Stage2 must NOT modify stage0 trainer.cpp; new training logic lives
//     in unsupervised_trainer.cu.
//   - Checkpoint must save complete d_synapses_ (32 MB incl. STDP state),
//     NOT just d_weights_ (4 MB).
//   - Reward mechanism is excluded from main path (B3 ablation only).
//     Trainer leaves dopamine at default 1.0 (no modulation).
// =============================================================================

// Run the unsupervised training loop for `n_steps` time steps.
// network must already be initialized via allocate_only() +
// init_columnar_synapses(). stream must already be loaded (file or fallback).
//
// On completion, returns the final step count (== start_step + n_steps).
// Returns -1 on error.
long run_unsupervised_training(
    SNNNetwork& network,
    TextStream& stream,
    long  start_step,        // typically 0; nonzero when resuming from ckpt
    long  n_steps,
    const std::string& ckpt_path,   // empty = no checkpointing
    bool  verbose           // print every STAGE2_LOG_INTERVAL steps
);

// Save full network state (d_synapses_ + d_row_ptr_ + d_col_idx_ + d_weights_)
// plus training metadata (step, text_pos) to `path`.
// Returns true on success.
bool save_checkpoint(
    SNNNetwork& network,
    const std::string& path,
    long step,
    size_t text_pos
);

// Load a checkpoint. Restores d_synapses_ etc. and returns the saved step
// and text_pos via out-params. Returns true on success.
// network must have been allocate_only()'d before calling this.
bool load_checkpoint(
    SNNNetwork& network,
    const std::string& path,
    long& out_step,
    size_t& out_text_pos
);

#endif // SNN_STAGE2_UNSUPERVISED_TRAINER_CUH
