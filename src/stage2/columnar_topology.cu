// =============================================================================
// columnar_topology.cu - stage 2 columnar topology generator (方案 A)
// =============================================================================
//
// Replaces stage0's uniform-random sparse connectivity with structured columns
// to fix the "signal dilution" failure mode documented in project memory:
//   "Three-layer uniform random sparse connectivity (10k neurons / 1M synapses)
//    cannot distinguish input patterns due to signal dilution and central
//    limit theorem effects."
//
// 方案 A: 真正的皮层柱架构
//   Layout: 10 columns x 1000 neurons = 10000 neurons
//   每柱内含三层流水线:
//     sensory 层:     [c*1000,       c*1000 + 200)   (200 neurons)
//     association 层: [c*1000 + 200, c*1000 + 800)   (600 neurons)
//     motor 层:       [c*1000 + 800, c*1000 + 1000) (200 neurons)
//
//   信号路径: 字节 b → 柱 (b%10) 的 sensory 层
//                  → intra-column → association 层
//                  → intra-column → motor 层
//                  → k-WTA 柱间竞争
//
// Each neuron's 100 synapses split as:
//   80 intra-column (pre in same column, may be any of S/A/M layer)
//     -- 这正是方案 A 的核心: 同柱突触跨层连接，信号在柱内三层间传播
//   20 inter-column (pre in other 9 columns)
//     -- 跨柱突触提供柱间信息交换和 k-WTA 竞争的基础
//
// 注意: columnar_topology 的连接生成逻辑与 stage2 v1 相同（按 idx/1000 分柱），
//       方案 A 的改动在 types.h (脑区划分) 和 text_stream.cu (输入注入)，
//       连接生成本身不需要改动，只需更新注释。
// =============================================================================

#include "columnar_topology.cuh"
#include "config.h"
#include "../include/synapse.cuh"   // sync_weights()
#include <curand_kernel.h>
#include <iostream>
#include <cuda_runtime.h>

// get_neuron_type() is __host__ __device__ inline in types.h, so usable here.

// -----------------------------------------------------------------------------
// Device kernel: one thread per post neuron. Generates 80 intra-column +
// 20 inter-column synapses for its assigned post neuron.
// -----------------------------------------------------------------------------
__global__ void init_columnar_synapses_kernel(
    Synapse*    synapses,
    int*        row_ptr,
    int*        col_idx,
    float*      weights,
    int         n_neurons,
    int         neurons_per_column,
    int         intra_per_neuron,
    int         inter_per_neuron,
    int         synapses_per_neuron,
    unsigned int seed
) {
    int post_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (post_idx >= n_neurons) return;

    int my_column      = post_idx / neurons_per_column;
    int intra_start    = my_column * neurons_per_column;
    int intra_size     = neurons_per_column;            // 1000

    int syn_start      = post_idx * synapses_per_neuron;
    row_ptr[post_idx]  = syn_start;

    curandState state;
    curand_init(seed, post_idx, 0, &state);

    // ----- 80 intra-column synapses -----
    for (int i = 0; i < intra_per_neuron; i++) {
        int syn_idx = syn_start + i;

        // Pick a random pre within the same column (excluding self).
        int pre_idx;
        do {
            pre_idx = intra_start
                    + (int)(curand_uniform(&state) * (float)intra_size);
        } while (pre_idx == post_idx || pre_idx >= intra_start + intra_size);

        bool pre_is_exc = (get_neuron_type(pre_idx) == NeuronType::EXCITATORY);
        float mag = curand_uniform(&state) * STAGE2_INIT_WEIGHT_MAX;
        float w   = pre_is_exc ? mag : -mag;

        synapses[syn_idx].pre_idx        = pre_idx;
        synapses[syn_idx].post_idx       = post_idx;
        synapses[syn_idx].weight         = w;
        synapses[syn_idx].delay          = STAGE2_SYNAPSE_DELAY;
        synapses[syn_idx].last_pre_spike  = -1000.0f;
        synapses[syn_idx].last_post_spike = -1000.0f;
        synapses[syn_idx].eligibility    = 0.0f;
        synapses[syn_idx]._pad           = 0.0f;

        col_idx[syn_idx] = pre_idx;
        weights[syn_idx] = w;
    }

    // ----- 20 inter-column synapses -----
    // Sample pre uniformly from the 9000 neurons NOT in my_column.
    // We use rejection sampling: draw a global index, reject if same column.
    for (int i = 0; i < inter_per_neuron; i++) {
        int syn_idx = syn_start + intra_per_neuron + i;

        int pre_idx;
        int attempts = 0;
        do {
            pre_idx = (int)(curand_uniform(&state) * (float)n_neurons);
            if (pre_idx >= n_neurons) pre_idx = n_neurons - 1;
            attempts++;
            // Safety: extremely unlikely to need >100 attempts with 9/10
            // acceptance probability, but guard anyway.
        } while (pre_idx / neurons_per_column == my_column && attempts < 100);

        // Fallback if rejection failed: walk forward until we cross a column
        if (pre_idx / neurons_per_column == my_column) {
            pre_idx = (my_column + 1) * neurons_per_column;
            if (pre_idx >= n_neurons) pre_idx = 0;
        }

        bool pre_is_exc = (get_neuron_type(pre_idx) == NeuronType::EXCITATORY);
        float mag = curand_uniform(&state) * STAGE2_INIT_WEIGHT_MAX;
        float w   = pre_is_exc ? mag : -mag;

        synapses[syn_idx].pre_idx        = pre_idx;
        synapses[syn_idx].post_idx       = post_idx;
        synapses[syn_idx].weight         = w;
        synapses[syn_idx].delay          = STAGE2_SYNAPSE_DELAY;
        synapses[syn_idx].last_pre_spike  = -1000.0f;
        synapses[syn_idx].last_post_spike = -1000.0f;
        synapses[syn_idx].eligibility    = 0.0f;
        synapses[syn_idx]._pad           = 0.0f;

        col_idx[syn_idx] = pre_idx;
        weights[syn_idx] = w;
    }

    // Last row_ptr entry: total synapse count
    if (post_idx == n_neurons - 1) {
        row_ptr[n_neurons] = n_neurons * synapses_per_neuron;
    }
}

// =============================================================================
// Host wrapper
// =============================================================================
void init_columnar_synapses(
    Synapse* d_synapses,
    int*     d_row_ptr,
    int*     d_col_idx,
    float*   d_weights,
    int      n_neurons,
    int      neurons_per_column,
    int      intra_per_neuron,
    int      inter_per_neuron,
    int      synapses_per_neuron,
    unsigned int seed
) {
    std::cout << "[Stage2-Topology] Generating columnar topology:" << std::endl;
    std::cout << "  neurons           = " << n_neurons << std::endl;
    std::cout << "  columns           = " << (n_neurons / neurons_per_column)
              << std::endl;
    std::cout << "  neurons/column    = " << neurons_per_column << std::endl;
    std::cout << "  intra/neuron      = " << intra_per_neuron << std::endl;
    std::cout << "  inter/neuron      = " << inter_per_neuron << std::endl;
    std::cout << "  total synapses    = "
              << (n_neurons * synapses_per_neuron) << std::endl;

    int blocks = (n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_columnar_synapses_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_synapses, d_row_ptr, d_col_idx, d_weights,
        n_neurons, neurons_per_column,
        intra_per_neuron, inter_per_neuron,
        synapses_per_neuron, seed);
    CUDA_CHECK_LAST();
    cudaDeviceSynchronize();
    std::cout << "[Stage2-Topology] Done." << std::endl;
}

// =============================================================================
// Diagnostic: verify topology on host (brief copy of synapse array)
// =============================================================================
void columnar_topology_report(
    const Synapse* d_synapses,
    int n_neurons,
    int neurons_per_column,
    int synapses_per_neuron
) {
    long total_syn = (long)n_neurons * synapses_per_neuron;
    Synapse* h_syn = new Synapse[total_syn];
    CUDA_CHECK(cudaMemcpy(h_syn, d_synapses,
                          total_syn * sizeof(Synapse),
                          cudaMemcpyDeviceToHost));

    long intra_count = 0;
    long inter_count = 0;
    long exc_count   = 0;
    long inh_count   = 0;
    long zero_count  = 0;
    double w_sum     = 0.0;
    double w_abs_sum = 0.0;

    for (long i = 0; i < total_syn; i++) {
        int pre  = h_syn[i].pre_idx;
        int post = h_syn[i].post_idx;
        float w  = h_syn[i].weight;
        if (w == 0.0f) { zero_count++; continue; }
        if (pre / neurons_per_column == post / neurons_per_column) {
            intra_count++;
        } else {
            inter_count++;
        }
        if (w > 0.0f) exc_count++; else inh_count++;
        w_sum     += w;
        w_abs_sum += (w < 0.0f ? -w : w);
    }

    std::cout << "[Stage2-Topology] Verification report:" << std::endl;
    std::cout << "  total synapses    = " << total_syn << std::endl;
    std::cout << "  nonzero           = " << (total_syn - zero_count)
              << " (zero=" << zero_count << ")" << std::endl;
    std::cout << "  intra-column      = " << intra_count
              << " (expected " << (long)n_neurons * INTRA_SYNAPSES_PER_NEURON
              << ")" << std::endl;
    std::cout << "  inter-column      = " << inter_count
              << " (expected " << (long)n_neurons * INTER_SYNAPSES_PER_NEURON
              << ")" << std::endl;
    std::cout << "  excitatory (w>0)  = " << exc_count << std::endl;
    std::cout << "  inhibitory (w<0)  = " << inh_count << std::endl;
    std::cout << "  mean weight       = " << (w_sum / total_syn) << std::endl;
    std::cout << "  mean |weight|     = " << (w_abs_sum / total_syn) << std::endl;

    delete[] h_syn;
}
