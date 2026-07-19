#ifndef SNN_STAGE2_COLUMNAR_TOPOLOGY_CUH
#define SNN_STAGE2_COLUMNAR_TOPOLOGY_CUH

#include "../include/types.h"

// =============================================================================
// Stage 2 columnar topology generator
// =============================================================================
// Replaces stage0's uniform-random sparse connectivity (network_init.cu) with
// a structured columnar layout:
//   - 10 columns x 1000 neurons (= N_TOTAL_NEURONS, unchanged)
//   - Each neuron grows 100 synapses:
//       * 80 intra-column  (dense local recurrence)
//       * 20 inter-column  (sparse long-range)
//   - Excitatory pre -> weight in [0, W_MAX]
//   - Inhibitory pre  -> weight in [-W_MAX, 0]
//   - Synapse layout matches stage0's CSR convention: row_ptr[post] points to
//     a contiguous block of `synapses_per_neuron` incoming synapses.
//
// This generator populates the SAME device buffers that SNNNetwork allocates
// in allocate_only(): d_synapses_, d_row_ptr_, d_col_idx_, d_weights_.
// After this call the caller must invoke stage0's sync_weights() -- actually
// not needed here because we set both d_synapses_.weight and d_weights_[i]
// to the same value at construction time.
// =============================================================================

// Host-side wrapper. Launches one CUDA thread per post neuron.
// n_neurons must equal N_TOTAL_NEURONS. synapses_per_neuron must equal
// SYNAPSES_PER_NEURON (= INTRA + INTER).
void init_columnar_synapses(
    Synapse* d_synapses,        // [n_neurons * synapses_per_neuron] (out)
    int*     d_row_ptr,         // [n_neurons + 1]                  (out)
    int*     d_col_idx,         // [n_neurons * synapses_per_neuron](out)
    float*   d_weights,         // [n_neurons * synapses_per_neuron](out)
    int      n_neurons,
    int      neurons_per_column,
    int      intra_per_neuron,
    int      inter_per_neuron,
    int      synapses_per_neuron,
    unsigned int seed
);

// Diagnostic: count how many synapses have nonzero weight, broken down by
// intra vs inter. Runs on host (copies arrays back briefly). Used by the
// stage 2a smoke test to verify topology was built correctly.
void columnar_topology_report(
    const Synapse* d_synapses,
    int n_neurons,
    int neurons_per_column,
    int synapses_per_neuron
);

#endif // SNN_STAGE2_COLUMNAR_TOPOLOGY_CUH
