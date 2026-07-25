// =============================================================================
// Stage 2e 显存分配器实现 (P0)
// =============================================================================

#include "memory_allocator.cuh"
#include <cstdio>

namespace stage2e {

template<typename T>
T* MemoryAllocator::alloc(size_t count, const char* name, size_t* accum_bytes) {
    size_t bytes = count * sizeof(T);
    T* ptr = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Stage2e] CUDA 分配失败: %s, 需要 %zu 字节 (%.2f MB)\n",
                name, bytes, bytes / (1024.0 * 1024.0));
        fprintf(stderr, "         错误: %s\n", cudaGetErrorString(err));
        allocation_failed_ = true;
        return nullptr;
    }
    // 清零 (避免 NaN 传播)
    cudaMemset(ptr, 0, bytes);
    *accum_bytes += bytes;
    vram_used_ += bytes;
    if (vram_used_ > vram_peak_) vram_peak_ = vram_used_;
    printf("  [+]. %-32s %10zu × %2zuB = %8.2f MB (累计 %7.2f MB)\n",
           name, count, sizeof(T), bytes / (1024.0 * 1024.0),
           vram_used_ / (1024.0 * 1024.0));
    return ptr;
}

size_t MemoryAllocator::allocate_all() {
    allocation_failed_ = false;
    printf("[Stage2e P0] 开始分配 GPU 显存...\n");
    printf("  %-34s %10s %8s %10s %10s\n",
           "Buffer", "Count", "Size", "Bytes", "Total MB");
    printf("  %-34s %10s %8s %10s %10s\n",
           "------", "-----", "----", "-----", "--------");

    size_t total = 0;

    // --- 神经元 ---
    d_bufs_.d_neurons           = alloc<NeuronStateAdEx>(N_TOTAL_NEURONS_2E, "d_neurons", &total);
    d_bufs_.d_spike_flags       = alloc<bool>(N_TOTAL_NEURONS_2E, "d_spike_flags", &total);

    // --- 突触结构 (CSR) ---
    d_bufs_.d_synapses          = alloc<BioSynapse>(N_TOTAL_SYNAPSES_2E, "d_synapses", &total);
    d_bufs_.d_csr_row_ptr       = alloc<int>(N_TOTAL_NEURONS_2E + 1, "d_csr_row_ptr", &total);
    d_bufs_.d_csr_col_idx       = alloc<int>(N_TOTAL_SYNAPSES_2E, "d_csr_col_idx", &total);
    d_bufs_.d_weights_cache     = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_weights_cache", &total);
    d_bufs_.d_eligibility       = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_eligibility", &total);
    d_bufs_.d_eligibility_slow  = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_eligibility_slow", &total);

    // --- PSW 概率突触权重 (10.7M × 4B × 2 = 85.6 MB) ---
    d_bufs_.d_synapse_alpha     = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_synapse_alpha (PSW)", &total);
    d_bufs_.d_synapse_beta      = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_synapse_beta (PSW)", &total);

    // --- v3 强化 A: PCA 反投影矩阵 (55K × 50 × 4B = 11 MB) ---
    d_bufs_.d_pca_W             = alloc<float>((size_t)N_TOTAL_NEURONS_2E * PATTERN_DIM,
                                               "d_pca_W (55K×50)", &total);

    // --- v3 强化 G: NMDA 钙浓度快照 ---
    d_bufs_.d_ca_snapshot       = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_ca_snapshot", &total);
    d_bufs_.d_ca_history_sparse = alloc<float>((size_t)CA_HISTORY_MAX_ACTIVE * CA_HISTORY_LEN,
                                               "d_ca_history_sparse", &total);

    // --- v4 强化 I: 突触传导延迟 ---
    d_bufs_.d_synapse_delay     = alloc<uint8_t>(N_TOTAL_SYNAPSES_2E, "d_synapse_delay", &total);
    // 延迟环形队列
    d_bufs_.d_delay_ring_indices = alloc<int>((size_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY,
                                              "d_delay_ring_indices", &total);
    d_bufs_.d_delay_ring_current = alloc<float>((size_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY,
                                                "d_delay_ring_current", &total);

    // --- v4 强化 J: STDP x_pre trace ---
    d_bufs_.d_stdp_x_pre_trace  = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_stdp_x_pre_trace", &total);
    d_bufs_.d_stdp_trace_epoch  = alloc<int>(N_TOTAL_SYNAPSES_2E, "d_stdp_trace_epoch (lazy)", &total);

    // --- v4 强化 K: CaMKII activity ---
    d_bufs_.d_camkii_activity   = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_camkii_activity", &total);

    // --- 输入电流 ---
    d_bufs_.d_input_current     = alloc<float>(N_TOTAL_NEURONS_2E, "d_input_current", &total);
    d_bufs_.d_nmda_current      = alloc<float>(N_TOTAL_NEURONS_2E, "d_nmda_current", &total);
    d_bufs_.d_nmda_post_state   = alloc<float2>(N_TOTAL_NEURONS_2E, "d_nmda_post_state", &total);
    d_bufs_.d_inhibitory_current = alloc<float>(N_TOTAL_NEURONS_2E, "d_inhibitory_current", &total);

    // --- 调质浓度 ---
    d_bufs_.d_da_concentration  = alloc<float>(N_TOTAL_NEURONS_2E, "d_da_concentration", &total);
    d_bufs_.d_ach_concentration = alloc<float>(N_TOTAL_NEURONS_2E, "d_ach_concentration", &total);
    d_bufs_.d_ne_concentration  = alloc<float>(N_TOTAL_NEURONS_2E, "d_ne_concentration", &total);
    d_bufs_.d_ht5_concentration = alloc<float>(N_TOTAL_NEURONS_2E, "d_ht5_concentration", &total);

    // --- v3 强化 C: 海马体索引 ---
    d_bufs_.d_hippo_indices     = alloc<HippoIndex>(HIPP_INDEX_SIZE, "d_hippo_indices", &total);

    // --- v3 强化 D: 共激活跟踪器 ---
    d_bufs_.d_coact_trackers    = alloc<CoactTracker>(COACT_TRACKER_SIZE, "d_coact_trackers", &total);

    // --- v3 强化 E: 工作记忆槽位 ---
    d_bufs_.d_wm_slots          = alloc<WMSlot>(WM_SLOTS, "d_wm_slots", &total);

    // --- DA 价值函数相关 ---
    d_bufs_.d_subcolumn_fr      = alloc<float>(W_VALUE_DIM, "d_subcolumn_fr", &total);
    d_bufs_.d_baseline_fr       = alloc<float>(W_VALUE_DIM, "d_baseline_fr", &total);
    d_bufs_.d_w_pred            = alloc<float>((size_t)W_PRED_DIM * W_PRED_DIM, "d_w_pred (200×200)", &total);
    d_bufs_.d_w_value           = alloc<float>(W_VALUE_DIM, "d_w_value", &total);
    d_bufs_.d_pred_fr           = alloc<float>(W_PRED_DIM, "d_pred_fr", &total);

    // --- 字节直方图 (NE 用) ---
    d_bufs_.d_byte_histogram    = alloc<int>(256, "d_byte_histogram", &total);

    // --- 卡方检验: 神经元×字节发放计数 (55K × 256 = 14.08M int32 = 56 MB) ---
    d_bufs_.d_neuron_byte_counts = alloc<int>(N_TOTAL_NEURONS_2E * 256,
                                              "d_neuron_byte_counts (55K×256)", &total);

    // --- 重放注入缓冲 ---
    d_bufs_.d_replay_injection  = alloc<float>(N_TOTAL_NEURONS_2E, "d_replay_injection", &total);

    printf("  %-34s %10s %8s %10s %10s\n",
           "------", "-----", "----", "-----", "--------");
    printf("[Stage2e P0] 持久显存分配完成: %.2f MB (%.2f GB)\n",
           vram_used_ / (1024.0 * 1024.0),
           vram_used_ / (1024.0 * 1024.0 * 1024.0));
    printf("[Stage2e P0] v4 设计目标: 1242 MB, 实际: %.2f MB, 偏差: %+.2f MB\n",
           vram_used_ / (1024.0 * 1024.0),
           vram_used_ / (1024.0 * 1024.0) - 1242.0);

    if (allocation_failed_) {
        fprintf(stderr, "[Stage2e P0] 一个或多个持久缓冲分配失败\n");
        return 0;
    }
    return vram_used_;
}

void MemoryAllocator::free_all() {
    if (vram_used_ == 0) return;

    printf("[Stage2e P0] 释放 GPU 显存 (%.2f MB)...\n",
           vram_used_ / (1024.0 * 1024.0));

    #define FREE_PTR(p) do { if (p) { cudaFree(p); p = nullptr; } } while(0)

    FREE_PTR(d_bufs_.d_neurons);
    FREE_PTR(d_bufs_.d_spike_flags);
    FREE_PTR(d_bufs_.d_synapses);
    FREE_PTR(d_bufs_.d_csr_row_ptr);
    FREE_PTR(d_bufs_.d_csr_col_idx);
    FREE_PTR(d_bufs_.d_weights_cache);
    FREE_PTR(d_bufs_.d_eligibility);
    FREE_PTR(d_bufs_.d_eligibility_slow);
    FREE_PTR(d_bufs_.d_synapse_alpha);
    FREE_PTR(d_bufs_.d_synapse_beta);
    FREE_PTR(d_bufs_.d_pca_W);
    FREE_PTR(d_bufs_.d_ca_snapshot);
    FREE_PTR(d_bufs_.d_ca_history_sparse);
    FREE_PTR(d_bufs_.d_synapse_delay);
    FREE_PTR(d_bufs_.d_delay_ring_indices);
    FREE_PTR(d_bufs_.d_delay_ring_current);
    FREE_PTR(d_bufs_.d_stdp_x_pre_trace);
    FREE_PTR(d_bufs_.d_stdp_trace_epoch);
    FREE_PTR(d_bufs_.d_camkii_activity);
    FREE_PTR(d_bufs_.d_input_current);
    FREE_PTR(d_bufs_.d_nmda_current);
    FREE_PTR(d_bufs_.d_nmda_post_state);
    FREE_PTR(d_bufs_.d_inhibitory_current);
    FREE_PTR(d_bufs_.d_da_concentration);
    FREE_PTR(d_bufs_.d_ach_concentration);
    FREE_PTR(d_bufs_.d_ne_concentration);
    FREE_PTR(d_bufs_.d_ht5_concentration);
    FREE_PTR(d_bufs_.d_hippo_indices);
    FREE_PTR(d_bufs_.d_coact_trackers);
    FREE_PTR(d_bufs_.d_wm_slots);
    FREE_PTR(d_bufs_.d_subcolumn_fr);
    FREE_PTR(d_bufs_.d_baseline_fr);
    FREE_PTR(d_bufs_.d_w_pred);
    FREE_PTR(d_bufs_.d_w_value);
    FREE_PTR(d_bufs_.d_pred_fr);
    FREE_PTR(d_bufs_.d_byte_histogram);
    FREE_PTR(d_bufs_.d_neuron_byte_counts);
    FREE_PTR(d_bufs_.d_replay_injection);

    #undef FREE_PTR

    vram_used_ = 0;
}

bool MemoryAllocator::check_budget() const {
    if (vram_used_ > budget_bytes_) {
        fprintf(stderr, "[Stage2e] 显存超预算! 已用 %.2f MB > 上限 %.2f MB\n",
                vram_used_ / (1024.0 * 1024.0),
                budget_bytes_ / (1024.0 * 1024.0));
        return false;
    }
    if (vram_used_ > budget_bytes_ * 9 / 10) {
        fprintf(stderr, "[Stage2e] 警告: 显存接近预算上限 (%.2f MB / %.2f MB)\n",
                vram_used_ / (1024.0 * 1024.0),
                budget_bytes_ / (1024.0 * 1024.0));
    }
    return true;
}

void MemoryAllocator::print_budget_report() const {
    printf("\n========== Stage 2e v4 显存预算报告 ==========\n");
    printf("  持久分配:       %.2f MB\n", vram_used_ / (1024.0 * 1024.0));
    printf("  峰值 (含临时):  %.2f MB\n", vram_peak_ / (1024.0 * 1024.0));
    printf("  v4 预算目标:    1332 MB\n");
    printf("  配置预算:       %.2f GB (%s)\n",
           budget_bytes_ / (1024.0 * 1024.0 * 1024.0),
           vram_used_ <= budget_bytes_ ? "OK" : "OVER");
    printf("  余量:           %.2f MB (%.1f%%)\n",
           budget_bytes_ > vram_used_ ? (budget_bytes_ - vram_used_) / (1024.0 * 1024.0) : 0.0,
           budget_bytes_ > vram_used_
               ? (budget_bytes_ - vram_used_) * 100.0 / budget_bytes_ : 0.0);
    printf("==============================================\n\n");

    // 查询 GPU 实际显存
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("  GPU 总显存:     %.2f GB\n", total_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  GPU 可用显存:   %.2f GB\n", free_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  本进程占用:     %.2f GB (%.1f%%)\n",
           (total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0),
           (total_bytes - free_bytes) * 100.0 / total_bytes);
}

} // namespace stage2e
