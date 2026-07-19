// =============================================================================
// competition.cu - stage 2d (P2) k-WTA columnar competition implementation
// =============================================================================
//
// Host-side implementation: N_COLUMNS=10 is small, so top-k selection is
// trivial on host. The only GPU ops are cudaMemcpy (D2H and H2D) of the
// d_spikes_ array (10KB total = N_TOTAL_NEURONS * sizeof(bool)).
//
// Per-step cost: 2 * 10KB memcpy + O(N) host scan = ~50us. Negligible vs.
// network.step() which takes ~600us.
// =============================================================================

#include "competition.cuh"
#include "config.h"
#include "../include/config.h"   // N_TOTAL_NEURONS, N_SENSORY_NEURONS, ...
#include "../include/types.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>

// Helper: compute column index for a neuron.
// Sensory neurons [0, N_SENSORY_NEURONS) are split across N_COLUMNS columns
// in round-robin fashion? No -- stage2's columnar_topology.cu assigns
// neurons to columns in contiguous blocks: column c owns neurons
// [c * NEURONS_PER_COLUMN, (c+1) * NEURONS_PER_COLUMN).
//
// BUT: stage0's brain region layout is
//   [0, N_SENSORY_NEURONS)            = sensory
//   [N_SENSORY_NEURONS, ...+ASSOC)    = association
//   [..., N_TOTAL_NEURONS)            = motor
//
// Stage2's columnar topology treats ALL 10000 neurons as a flat array
// divided into 10 columns of 1000 neurons each. So column c owns
// [c * 1000, (c+1) * 1000). This crosses brain region boundaries but
// that's fine -- columnar_topology.cu's init_columnar_synapses_kernel
// uses exactly this layout (verified in 2a smoke test: 10 columns x 1000
// neurons, 80 intra + 20 inter per neuron).
//
// So: neuron n belongs to column (n / NEURONS_PER_COLUMN).
static inline int column_of(int neuron_id) {
    return neuron_id / NEURONS_PER_COLUMN;
}

void apply_kwta_competition(SNNNetwork& network, int k) {
    // Clamp k to valid range
    if (k < 1) k = 1;
    if (k > N_COLUMNS) k = N_COLUMNS;

    const int N = N_TOTAL_NEURONS;

    // 1. Pull d_spikes_ to host
    bool* h_spikes = new bool[N];
    CUDA_CHECK(cudaMemcpy(h_spikes, network.get_d_spikes(),
                          N * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    // 2. Count spikes per column
    long col_counts[N_COLUMNS];
    std::memset(col_counts, 0, sizeof(col_counts));
    for (int n = 0; n < N; n++) {
        if (h_spikes[n]) {
            col_counts[column_of(n)]++;
        }
    }

    // 3. Find top-k columns by spike count.
    //    Use a simple index sort (N_COLUMNS=10 is tiny).
    int col_idx[N_COLUMNS];
    for (int c = 0; c < N_COLUMNS; c++) col_idx[c] = c;
    std::sort(col_idx, col_idx + N_COLUMNS,
              [&](int a, int b) { return col_counts[a] > col_counts[b]; });

    // Build a "winner" mask
    bool is_winner[N_COLUMNS] = {false};
    for (int i = 0; i < k; i++) {
        is_winner[col_idx[i]] = true;
    }

    // 4. Zero out spikes in non-winner columns
    for (int n = 0; n < N; n++) {
        if (h_spikes[n] && !is_winner[column_of(n)]) {
            h_spikes[n] = false;
        }
    }

    // 5. Upload filtered spikes back to device
    CUDA_CHECK(cudaMemcpy(network.get_d_spikes(), h_spikes,
                          N * sizeof(bool),
                          cudaMemcpyHostToDevice));

    delete[] h_spikes;
}
