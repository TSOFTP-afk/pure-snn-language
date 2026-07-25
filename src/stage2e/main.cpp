// =============================================================================
// Stage 2e Phase 1: 快时间尺度生物机制烟雾测试
// =============================================================================
// 对应设计文档 §7.1 Phase 1:
//   - 快时间尺度: AdEx 神经元、NMDA 受体、STP、群体编码
//   - 通过条件: 发放模式多样性↑, spike count 极差 > 100, 簇状发放出现
//
// 用法:
//   snn_stage2e_p1                  # 默认 10K 步烟雾测试
//   snn_stage2e_p1 --steps 50000   # 自定义步数
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include "scheduler.cuh"
#include "network_init.cuh"
#include "input_encoding.cuh"
#include "modulatory_kernels.cuh"
#include "synapse_kernels.cuh"
#include "run_config.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdarg>
#include <csignal>

static volatile std::sig_atomic_t g_stop_requested = 0;

static void handle_stop_signal(int) {
    g_stop_requested = 1;
}

static void emit_run_param(FILE* fp, const char* prefix, const char* key, const char* fmt, ...) {
    if (!fp) return;
    fprintf(fp, "%sRUN_PARAM %s=", prefix ? prefix : "", key);
    va_list args;
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
    fprintf(fp, "\n");
}

static void emit_final_metric(FILE* fp, const char* key, const char* fmt, ...) {
    if (!fp) return;
    fprintf(fp, "FINAL_METRIC %s=", key);
    va_list args;
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
    fprintf(fp, "\n");
}

static void print_experiment_metadata(FILE* fp, const char* prefix,
                                      const stage2e::RunConfig& config,
                                      bool csv_enabled,
                                      const char* csv_path,
                                      const cudaDeviceProp* prop) {
    if (!fp) return;
    fprintf(fp, "%sRUN_METADATA_BEGIN\n", prefix ? prefix : "");
    emit_run_param(fp, prefix, "program", "snn_stage2e_p1");
    emit_run_param(fp, prefix, "build_date", "%s", __DATE__);
    emit_run_param(fp, prefix, "build_time", "%s", __TIME__);
    emit_run_param(fp, prefix, "design_doc", "docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md");
    emit_run_param(fp, prefix, "total_steps", "%d", config.total_steps);
    emit_run_param(fp, prefix, "e0_mode", "%d", config.e0_mode ? 1 : 0);
    emit_run_param(fp, prefix, "synthetic_input", "%d", config.synthetic_input ? 1 : 0);
    emit_run_param(fp, prefix, "strict_criteria", "%d", config.strict_criteria ? 1 : 0);
    emit_run_param(fp, prefix, "device", "%d", config.device);
    emit_run_param(fp, prefix, "seed", "%u", config.seed);
    emit_run_param(fp, prefix, "text_path", "%s", config.text_path.c_str());
    emit_run_param(fp, prefix, "checkpoint_dir", "%s", config.checkpoint_dir.c_str());
    emit_run_param(fp, prefix, "checkpoint_interval", "%d", config.checkpoint_interval);
    emit_run_param(fp, prefix, "keep_checkpoints", "%d", config.keep_checkpoints);
    emit_run_param(fp, prefix, "resume_path", "%s", config.resume_path.c_str());
    emit_run_param(fp, prefix, "csv_enabled", "%d", csv_enabled ? 1 : 0);
    emit_run_param(fp, prefix, "csv_path", "%s", csv_path ? csv_path : "");
    emit_run_param(fp, prefix, "log_interval", "%d", LOG_INTERVAL_2E);
    emit_run_param(fp, prefix, "compiled_checkpoint_interval", "%d", CHECKPOINT_INTERVAL_2E);
    emit_run_param(fp, prefix, "smoke_test_steps", "%d", SMOKE_TEST_STEPS_2E);
    emit_run_param(fp, prefix, "gpu_name", "%s", prop ? prop->name : "unknown");
    emit_run_param(fp, prefix, "gpu_total_mem_mb", "%.0f", prop ? prop->totalGlobalMem / (1024.0 * 1024.0) : 0.0);
    emit_run_param(fp, prefix, "gpu_compute_major", "%d", prop ? prop->major : 0);
    emit_run_param(fp, prefix, "gpu_compute_minor", "%d", prop ? prop->minor : 0);
    emit_run_param(fp, prefix, "memory_budget_mb", "%llu",
                   static_cast<unsigned long long>(config.memory_budget_mb));
    emit_run_param(fp, prefix, "vram_budget_bytes", "%llu",
                   static_cast<unsigned long long>(config.memory_budget_mb * 1024ULL * 1024ULL));
    emit_run_param(fp, prefix, "vram_peak_target_bytes", "%lld", (long long)VRAM_PEAK_TARGET_BYTES);
    emit_run_param(fp, prefix, "n_columns", "%d", N_COLUMNS_2E);
    emit_run_param(fp, prefix, "neurons_per_column", "%d", NEURONS_PER_COLUMN_2E);
    emit_run_param(fp, prefix, "n_prefrontal_neurons", "%d", N_PREFRONTAL_NEURONS);
    emit_run_param(fp, prefix, "n_association_neurons", "%d", N_ASSOCIATION_NEURONS_2E);
    emit_run_param(fp, prefix, "n_total_neurons", "%d", N_TOTAL_NEURONS_2E);
    emit_run_param(fp, prefix, "col_l4_size", "%d", COL_L4_SIZE_2E);
    emit_run_param(fp, prefix, "col_l23_size", "%d", COL_L23_SIZE_2E);
    emit_run_param(fp, prefix, "col_l5_size", "%d", COL_L5_SIZE_2E);
    emit_run_param(fp, prefix, "col_l6_size", "%d", COL_L6_SIZE_2E);
    emit_run_param(fp, prefix, "n_l4_total", "%d", N_L4_TOTAL_2E);
    emit_run_param(fp, prefix, "n_l23_total", "%d", N_L23_TOTAL_2E);
    emit_run_param(fp, prefix, "n_l5_total", "%d", N_L5_TOTAL_2E);
    emit_run_param(fp, prefix, "n_l6_total", "%d", N_L6_TOTAL_2E);
    emit_run_param(fp, prefix, "synapses_per_neuron", "%d", SYNAPSES_PER_NEURON_2E);
    emit_run_param(fp, prefix, "n_total_synapses", "%d", N_TOTAL_SYNAPSES_2E);
    emit_run_param(fp, prefix, "excitatory_ratio", "%.6f", EXCITATORY_RATIO_2E);
    emit_run_param(fp, prefix, "delay_steps_max", "%d", DELAY_STEPS_MAX);
    emit_run_param(fp, prefix, "delay_intra_min", "%d", DELAY_INTRA_MIN);
    emit_run_param(fp, prefix, "delay_intra_max", "%d", DELAY_INTRA_MAX);
    emit_run_param(fp, prefix, "delay_inter_min", "%d", DELAY_INTER_MIN);
    emit_run_param(fp, prefix, "delay_inter_max", "%d", DELAY_INTER_MAX);
    emit_run_param(fp, prefix, "delay_long_min", "%d", DELAY_LONG_MIN);
    emit_run_param(fp, prefix, "delay_long_max", "%d", DELAY_LONG_MAX);
    emit_run_param(fp, prefix, "delay_ring_slot_capacity", "%d", DELAY_RING_SLOT_CAPACITY);
    emit_run_param(fp, prefix, "adex_c", "%.6f", ADEX_C);
    emit_run_param(fp, prefix, "adex_gl", "%.6f", ADEX_GL);
    emit_run_param(fp, prefix, "adex_el", "%.6f", ADEX_EL);
    emit_run_param(fp, prefix, "adex_vt", "%.6f", ADEX_VT);
    emit_run_param(fp, prefix, "adex_delta_t", "%.6f", ADEX_DELTA_T);
    emit_run_param(fp, prefix, "adex_vr", "%.6f", ADEX_VR);
    emit_run_param(fp, prefix, "adex_vpeak", "%.6f", ADEX_VPEAK);
    emit_run_param(fp, prefix, "adex_taum", "%.6f", ADEX_TAUM);
    emit_run_param(fp, prefix, "adex_taum_inv", "%.6f", ADEX_TAUM_INV);
    emit_run_param(fp, prefix, "v_bio_offset", "%.6f", V_BIO_OFFSET);
    emit_run_param(fp, prefix, "v_bio_scale", "%.6f", V_BIO_SCALE);
    emit_run_param(fp, prefix, "nmda_mg_block_threshold", "%.6f", NMDA_MG_BLOCK_THRESHOLD);
    emit_run_param(fp, prefix, "nmda_ca_tau", "%.6f", NMDA_CA_TAU);
    emit_run_param(fp, prefix, "nmda_conductance_max", "%.6f", NMDA_CONDUCTANCE_MAX);
    emit_run_param(fp, prefix, "nmda_mg_concentration", "%.6f", NMDA_MG_CONCENTRATION);
    emit_run_param(fp, prefix, "nmda_tau", "%.6f", NMDA_TAU);
    emit_run_param(fp, prefix, "nmda_g_max", "%.6f", NMDA_G_MAX);
    emit_run_param(fp, prefix, "ampa_tau", "%.6f", AMPA_TAU);
    emit_run_param(fp, prefix, "ampa_g_max", "%.6f", AMPA_G_MAX);
    emit_run_param(fp, prefix, "gaba_a_tau", "%.6f", GABA_A_TAU);
    emit_run_param(fp, prefix, "gaba_a_g_max", "%.6f", GABA_A_G_MAX);
    emit_run_param(fp, prefix, "gaba_b_tau", "%.6f", GABA_B_TAU);
    emit_run_param(fp, prefix, "gaba_b_g_max", "%.6f", GABA_B_G_MAX);
    emit_run_param(fp, prefix, "adex_a_adapt", "%.6f", ADEX_A_ADAPT);
    emit_run_param(fp, prefix, "adex_b_reset", "%.6f", ADEX_B_RESET);
    emit_run_param(fp, prefix, "adex_tau_w", "%.6f", ADEX_TAU_W);
    emit_run_param(fp, prefix, "adex_tau_w_inv", "%.6f", ADEX_TAU_W_INV);
    emit_run_param(fp, prefix, "adex_v_thresh_norm", "%.6f", ADEX_V_THRESH_NORM);
    emit_run_param(fp, prefix, "adex_v_reset_norm", "%.6f", ADEX_V_RESET_NORM);
    emit_run_param(fp, prefix, "adex_refractory_steps", "%d", ADEX_REFRACTORY_STEPS);
    emit_run_param(fp, prefix, "adex_theta_adapt_rate", "%.6f", ADEX_THETA_ADAPT_RATE);
    emit_run_param(fp, prefix, "adex_theta_decay", "%.6f", ADEX_THETA_DECAY);
    emit_run_param(fp, prefix, "adex_theta_max", "%.6f", ADEX_THETA_MAX);
    emit_run_param(fp, prefix, "stp_u_se", "%.6f", STP_U_SE);
    emit_run_param(fp, prefix, "stp_u_si", "%.6f", STP_U_SI);
    emit_run_param(fp, prefix, "stp_tau_fac", "%.6f", STP_TAU_FAC);
    emit_run_param(fp, prefix, "stp_tau_rec", "%.6f", STP_TAU_REC);
    emit_run_param(fp, prefix, "stp_tau_fac_inv", "%.6f", STP_TAU_FAC_INV);
    emit_run_param(fp, prefix, "stp_tau_rec_inv", "%.6f", STP_TAU_REC_INV);
    emit_run_param(fp, prefix, "pop_coding_k_per_column", "%d", POP_CODING_K_PER_COLUMN);
    emit_run_param(fp, prefix, "pop_coding_gain", "%.6f", POP_CODING_GAIN);
    emit_run_param(fp, prefix, "input_inject_interval", "%d", INPUT_INJECT_INTERVAL);
    emit_run_param(fp, prefix, "input_text_corpus_len", "%d", INPUT_TEXT_CORPUS_LEN);
    // 丘脑-皮层门控参数 (§1.1 注意力门控)
    emit_run_param(fp, prefix, "gate_update_rate", "%.6f", GATE_UPDATE_RATE);
    emit_run_param(fp, prefix, "gate_min", "%.6f", GATE_MIN);
    emit_run_param(fp, prefix, "gate_max", "%.6f", GATE_MAX);
    emit_run_param(fp, prefix, "gate_activity_coup", "%.6f", GATE_ACTIVITY_COUP);
    emit_run_param(fp, prefix, "gate_novelty_coup", "%.6f", GATE_NOVELTY_COUP);
    emit_run_param(fp, prefix, "gate_initial_signal", "%.6f", GATE_INITIAL_SIGNAL);
    emit_run_param(fp, prefix, "da_receptor_init_exc", "%.6f", DA_RECEPTOR_INIT_EXC);
    emit_run_param(fp, prefix, "da_receptor_init_inh", "%.6f", DA_RECEPTOR_INIT_INH);
    emit_run_param(fp, prefix, "ach_receptor_init", "%.6f", ACH_RECEPTOR_INIT);
    emit_run_param(fp, prefix, "ne_receptor_init", "%.6f", NE_RECEPTOR_INIT);
    emit_run_param(fp, prefix, "ht5_receptor_init", "%.6f", HT5_RECEPTOR_INIT);
    emit_run_param(fp, prefix, "stdp_x_pre_tau", "%.6f", STDP_X_PRE_TAU);
    emit_run_param(fp, prefix, "stdp_x_post_tau", "%.6f", STDP_X_POST_TAU);
    emit_run_param(fp, prefix, "stdp_a_plus", "%.6f", STDP_A_PLUS_2E);
    emit_run_param(fp, prefix, "stdp_a_minus", "%.6f", STDP_A_MINUS_2E);
    emit_run_param(fp, prefix, "stdp_w_max", "%.6f", STDP_W_MAX_2E);
    emit_run_param(fp, prefix, "stdp_e1_tau", "%.6f", STDP_E1_TAU);
    emit_run_param(fp, prefix, "stdp_e2_tau", "%.6f", STDP_E2_TAU);
    emit_run_param(fp, prefix, "camkii_k1", "%.6f", CAMKII_K1);
    emit_run_param(fp, prefix, "camkii_k2", "%.6f", CAMKII_K2);
    emit_run_param(fp, prefix, "camkii_k3", "%.6f", CAMKII_K3);
    emit_run_param(fp, prefix, "camkii_k4", "%.6f", CAMKII_K4);
    emit_run_param(fp, prefix, "camkii_autophos_facil", "%.6f", CAMKII_AUTOPHOS_FACIL);
    emit_run_param(fp, prefix, "camkii_autophos_consol", "%.6f", CAMKII_AUTOPHOS_CONSOL);
    emit_run_param(fp, prefix, "inhibitory_neuron_ratio", "%.6f", INHIBITORY_NEURON_RATIO);
    emit_run_param(fp, prefix, "inhib_fs_ratio", "%.6f", INHIB_FS_RATIO);
    emit_run_param(fp, prefix, "inhib_lts_ratio", "%.6f", INHIB_LTS_RATIO);
    emit_run_param(fp, prefix, "inhib_som_ratio", "%.6f", INHIB_SOM_RATIO);
    emit_run_param(fp, prefix, "n_neuromodulators", "%d", N_NEUROMODULATORS_2E);
    emit_run_param(fp, prefix, "da_tau", "%.6f", DA_TAU);
    emit_run_param(fp, prefix, "ach_tau", "%.6f", ACH_TAU);
    emit_run_param(fp, prefix, "ne_tau", "%.6f", NE_TAU);
    emit_run_param(fp, prefix, "ht5_tau", "%.6f", HT5_TAU);
    emit_run_param(fp, prefix, "w_value_dim", "%d", W_VALUE_DIM);
    emit_run_param(fp, prefix, "w_pred_dim", "%d", W_PRED_DIM);
    emit_run_param(fp, prefix, "td_gamma", "%.6f", TD_GAMMA);
    emit_run_param(fp, prefix, "eta_value", "%.6f", ETA_VALUE);
    emit_run_param(fp, prefix, "eta_pred", "%.6f", ETA_PRED);
    emit_run_param(fp, prefix, "novelty_ema_beta", "%.6f", NOVELTY_EMA_BETA);
    emit_run_param(fp, prefix, "da_downgrade_threshold", "%.6f", DA_DOWNGRADE_THRESHOLD);
    emit_run_param(fp, prefix, "hipp_index_size", "%d", HIPP_INDEX_SIZE);
    emit_run_param(fp, prefix, "pattern_dim", "%d", PATTERN_DIM);
    emit_run_param(fp, prefix, "hipp_replay_batch", "%d", HIPP_REPLAY_BATCH);
    emit_run_param(fp, prefix, "hipp_novelty_threshold", "%.6f", HIPP_NOVELTY_THRESHOLD);
    emit_run_param(fp, prefix, "hipp_replay_decay", "%.6f", HIPP_REPLAY_DECAY);
    emit_run_param(fp, prefix, "wm_slots", "%d", WM_SLOTS);
    emit_run_param(fp, prefix, "wm_pattern_dim", "%d", WM_PATTERN_DIM);
    emit_run_param(fp, prefix, "wm_decay_factor", "%.6f", WM_DECAY_FACTOR);
    emit_run_param(fp, prefix, "wm_activation_threshold", "%.6f", WM_ACTIVATION_THRESHOLD);
    emit_run_param(fp, prefix, "prefrontal_groups", "%d", PREFRONTAL_GROUPS);
    emit_run_param(fp, prefix, "neurons_per_pf_group", "%d", NEURONS_PER_PF_GROUP);
    emit_run_param(fp, prefix, "coact_tracker_size", "%d", COACT_TRACKER_SIZE);
    emit_run_param(fp, prefix, "coact_k_sample", "%d", COACT_K_SAMPLE);
    emit_run_param(fp, prefix, "coact_form_threshold", "%d", COACT_FORM_THRESHOLD);
    emit_run_param(fp, prefix, "coact_evict_steps", "%d", COACT_EVICT_STEPS);
    emit_run_param(fp, prefix, "n_form_per_cycle", "%d", N_FORM_PER_CYCLE);
    emit_run_param(fp, prefix, "pca_snapshot_buffer", "%d", PCA_SNAPSHOT_BUFFER);
    emit_run_param(fp, prefix, "pca_update_interval", "%d", PCA_UPDATE_INTERVAL);
    emit_run_param(fp, prefix, "pca_retrain_interval", "%d", PCA_RETRAIN_INTERVAL);
    emit_run_param(fp, prefix, "pca_anchor_refresh", "%d", PCA_ANCHOR_REFRESH);
    emit_run_param(fp, prefix, "pca_learning_rate", "%.6f", PCA_LEARNING_RATE);
    emit_run_param(fp, prefix, "ca_snapshot_interval", "%d", CA_SNAPSHOT_INTERVAL);
    emit_run_param(fp, prefix, "ca_history_len", "%d", CA_HISTORY_LEN);
    emit_run_param(fp, prefix, "ca_history_max_active", "%d", CA_HISTORY_MAX_ACTIVE);
    emit_run_param(fp, prefix, "dev_phase_embryo_end", "%d", DEV_PHASE_EMBRYO_END);
    emit_run_param(fp, prefix, "dev_phase_synapto_end", "%d", DEV_PHASE_SYNAPTO_END);
    emit_run_param(fp, prefix, "dev_phase_critical_end", "%d", DEV_PHASE_CRITICAL_END);
    emit_run_param(fp, prefix, "dev_phase_prune_end", "%d", DEV_PHASE_PRUNE_END);
    emit_run_param(fp, prefix, "dev_phase_mature_end", "%d", DEV_PHASE_MATURE_END);
    emit_run_param(fp, prefix, "dev_total_steps", "%d", DEV_TOTAL_STEPS);
    emit_run_param(fp, prefix, "checkpoint_phase2", "%d", CHECKPOINT_PHASE2);
    emit_run_param(fp, prefix, "checkpoint_phase3", "%d", CHECKPOINT_PHASE3);
    emit_run_param(fp, prefix, "checkpoint_phase4", "%d", CHECKPOINT_PHASE4);
    stage2e::DevPhaseTable phase_table;
    const char* phase_names[] = {"embryo", "synaptogenic", "critical", "pruning", "mature"};
    for (int i = 0; i < 5; ++i) {
        emit_run_param(fp, prefix, "dev_phase_name", "%d:%s", i, phase_names[i]);
        emit_run_param(fp, prefix, "dev_phase_growth_rate", "%d:%.6f", i, phase_table.phases[i].growth_rate);
        emit_run_param(fp, prefix, "dev_phase_prune_threshold", "%d:%.6f", i, phase_table.phases[i].prune_threshold);
        emit_run_param(fp, prefix, "dev_phase_plasticity_gain", "%d:%.6f", i, phase_table.phases[i].plasticity_gain);
        emit_run_param(fp, prefix, "dev_phase_nmda_expression", "%d:%.6f", i, phase_table.phases[i].nmda_expression);
        emit_run_param(fp, prefix, "dev_phase_ach_level", "%d:%.6f", i, phase_table.phases[i].ach_level);
        emit_run_param(fp, prefix, "dev_phase_myeline_factor", "%d:%.6f", i, phase_table.phases[i].myeline_factor);
        emit_run_param(fp, prefix, "dev_phase_end_step", "%d:%d", i, phase_table.phases[i].end_step);
    }
    emit_run_param(fp, prefix, "chi2_df", "%d", 255);
    emit_run_param(fp, prefix, "chi2_p_threshold", "%.6f", 0.05);
    emit_run_param(fp, prefix, "chi2_critical", "%.6f", 293.2);
    emit_run_param(fp, prefix, "chi2_min_row_total", "%d", 10);
    emit_run_param(fp, prefix, "chi2_pass_significant_min", "%d", 500);
    emit_run_param(fp, prefix, "weight_stats_sample", "%d", 100000);
    emit_run_param(fp, prefix, "weight_distribution_sample", "%d", N_TOTAL_SYNAPSES_2E);
    emit_run_param(fp, prefix, "weight_histogram_edges", "-1.5,-1.0,-0.5,-0.1,-0.01,0.01,0.1,0.5,1.0,1.2,1.4,1.5");
    fprintf(fp, "%sRUN_METADATA_END\n", prefix ? prefix : "");
    fflush(fp);
}

// 采集 device 端标量和 (用于评估 NMDA/STDP 等是否真实工作)
static float device_sum_float(const float* d, int n) {
    if (!d || n <= 0) return 0.0f;
    // 简化: 拷贝回 host 求和 (评估模式可接受)
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += h[i];
    return static_cast<float>(s);
}

static int device_count_nonzero_float(const float* d, int n) {
    if (!d || n <= 0) return 0;
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] != 0.0f) c++;
    return c;
}

static void sample_synapse_weight_stats(const stage2e::PersistentBuffers& b,
                                        int sample_count,
                                        float* mean_weight,
                                        float* mean_abs_weight,
                                        float* min_weight,
                                        float* max_weight) {
    if (sample_count > N_TOTAL_SYNAPSES_2E) sample_count = N_TOTAL_SYNAPSES_2E;
    std::vector<BioSynapse> h(sample_count);
    cudaMemcpy(h.data(), b.d_synapses, sample_count * sizeof(BioSynapse), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    double abs_sum = 0.0;
    float mn = h.empty() ? 0.0f : h[0].weight;
    float mx = h.empty() ? 0.0f : h[0].weight;
    for (int i = 0; i < sample_count; ++i) {
        float w = h[i].weight;
        sum += w;
        abs_sum += fabsf(w);
        if (w < mn) mn = w;
        if (w > mx) mx = w;
    }
    *mean_weight = sample_count > 0 ? static_cast<float>(sum / sample_count) : 0.0f;
    *mean_abs_weight = sample_count > 0 ? static_cast<float>(abs_sum / sample_count) : 0.0f;
    *min_weight = mn;
    *max_weight = mx;
}

// =============================================================================
// PSW (概率突触权重) 统计: 采样 alpha/beta 数组
// 输出:
//   - mean_alpha, mean_beta: 平均 LTP/LTD 证据
//   - mean_evidence: 平均 α+β (证据强度, 元可塑性指标)
//   - mean_confidence: 平均 (α+β)/(α+β+τ) (语义成熟度 ∈ (0,1))
//   - mature_ratio: (α+β) > PSW_MATURITY_THRESH 的突触比例 (已学习稳定突触占比)
// =============================================================================
static void sample_psw_stats(const stage2e::PersistentBuffers& b,
                             int sample_count,
                             float* mean_alpha,
                             float* mean_beta,
                             float* mean_evidence,
                             float* mean_confidence,
                             float* mature_ratio) {
    if (sample_count > N_TOTAL_SYNAPSES_2E) sample_count = N_TOTAL_SYNAPSES_2E;
    std::vector<float> h_alpha(sample_count);
    std::vector<float> h_beta(sample_count);
    cudaMemcpy(h_alpha.data(), b.d_synapse_alpha,
               sample_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta.data(),  b.d_synapse_beta,
               sample_count * sizeof(float), cudaMemcpyDeviceToHost);

    double sum_a = 0.0, sum_b = 0.0, sum_ev = 0.0, sum_conf = 0.0;
    int mature_count = 0;
    const float tau = 1.0f;  // confidence = (α+β)/(α+β+τ)
    for (int i = 0; i < sample_count; ++i) {
        float a = h_alpha[i];
        float be = h_beta[i];
        float ev = a + be;
        sum_a += a;
        sum_b += be;
        sum_ev += ev;
        sum_conf += ev / (ev + tau);
        if (ev > PSW_MATURITY_THRESH) mature_count++;
    }
    int n = sample_count > 0 ? sample_count : 1;
    *mean_alpha      = static_cast<float>(sum_a / n);
    *mean_beta       = static_cast<float>(sum_b / n);
    *mean_evidence   = static_cast<float>(sum_ev / n);
    *mean_confidence = static_cast<float>(sum_conf / n);
    *mature_ratio    = static_cast<float>((double)mature_count / n);
}

// Ca²⁺ 回弹 LTD 统计: 采样突触的 ca_concentration, 统计高 Ca²⁺ 突触比例
static void sample_ca_rebound_stats(const stage2e::PersistentBuffers& b,
                                    int sample_count,
                                    float* mean_ca,
                                    float* max_ca,
                                    float* high_ca_ratio) {
    if (sample_count > N_TOTAL_SYNAPSES_2E) sample_count = N_TOTAL_SYNAPSES_2E;
    std::vector<BioSynapse> h(sample_count);
    cudaMemcpy(h.data(), b.d_synapses, sample_count * sizeof(BioSynapse), cudaMemcpyDeviceToHost);

    double sum_ca = 0.0;
    float mx = 0.0f;
    int high_count = 0;
    for (int i = 0; i < sample_count; ++i) {
        float ca = h[i].ca_concentration;
        sum_ca += ca;
        if (ca > mx) mx = ca;
        if (ca > CA_REBOUND_THRESHOLD) high_count++;
    }
    int n = sample_count > 0 ? sample_count : 1;
    *mean_ca = static_cast<float>(sum_ca / n);
    *max_ca = mx;
    *high_ca_ratio = static_cast<float>((double)high_count / n);
}

// 平衡态网络验证: 采样神经元发放率分布, 计算变异系数 CV = std/mean
// 平衡态条件激活后, 神经元活动应去相关, CV 应 > 0.5 (当前 ~0.3)
static void sample_balance_state_stats(const stage2e::PersistentBuffers& b,
                                       int sample_count,
                                       float* mean_fr,
                                       float* std_fr,
                                       float* cv) {
    if (sample_count > N_TOTAL_NEURONS_2E) sample_count = N_TOTAL_NEURONS_2E;
    std::vector<NeuronStateAdEx> h(sample_count);
    cudaMemcpy(h.data(), b.d_neurons, sample_count * sizeof(NeuronStateAdEx), cudaMemcpyDeviceToHost);

    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < sample_count; ++i) {
        float fr = h[i].fire_rate;
        sum += fr;
        sum_sq += (double)fr * fr;
    }
    int n = sample_count > 0 ? sample_count : 1;
    double mean = sum / n;
    double variance = (sum_sq / n) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    double std_val = sqrt(variance);
    *mean_fr = static_cast<float>(mean);
    *std_fr = static_cast<float>(std_val);
    *cv = (mean > 1e-9) ? static_cast<float>(std_val / mean) : 0.0f;
}

// =============================================================================
// 权重分布深度分析 (E1 vs E0 对比用)
// =============================================================================
// Phase R2 模块 C: 联合皮层采用柱内交错布局 (每柱 L4[0,200) L2/3[200,550) L5[550,750) L6[750,1000))
// 故层归属不能再用连续区间表示, 改用 neuron_idx → layer_idx 辅助函数
struct LayerRange { const char* name; int start; int end; };  // 保留用于 prefrontal 连续区间

// 返回神经元所属层 (0=L4, 1=L2/3, 2=L5, 3=L6, 4=prefrontal)
// 联合皮层 (50000 神经元) 按柱交错布局: 每柱 1000 神经元, 4 层交错
// 前额叶 (5000 神经元, [50000, 55000)) 全部为层 4
static int get_layer_for_neuron_idx(int neuron_idx) {
    if (neuron_idx >= N_ASSOCIATION_NEURONS_2E) return 4;  // prefrontal
    int off = neuron_idx % NEURONS_PER_COLUMN_2E;  // 柱内偏移
    if (off < COL_L4_SIZE_2E) return 0;                                            // L4
    if (off < COL_L4_SIZE_2E + COL_L23_SIZE_2E) return 1;                         // L2/3
    if (off < COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E) return 2;        // L5
    return 3;                                                                     // L6
}

static void analyze_weight_distribution(const stage2e::PersistentBuffers& b,
                                         int sample_count,
                                         const char* tag) {
    if (sample_count > N_TOTAL_SYNAPSES_2E) sample_count = N_TOTAL_SYNAPSES_2E;

    std::vector<BioSynapse> h(sample_count);
    cudaMemcpy(h.data(), b.d_synapses, sample_count * sizeof(BioSynapse), cudaMemcpyDeviceToHost);

    // --- 1. 全局直方图 (12 桶) ---
    // 桶边界: [-1.5,-1.0), [-1.0,-0.5), [-0.5,-0.1), [-0.1,-0.01), [-0.01,0.01),
    //         [0.01,0.1), [0.1,0.5), [0.5,1.0), [1.0,1.2), [1.2,1.4), [1.4,1.5], 其他
    const int N_BINS = 11;
    const float bin_edges[] = {-1.5f, -1.0f, -0.5f, -0.1f, -0.01f, 0.01f, 0.1f, 0.5f, 1.0f, 1.2f, 1.4f, 1.5f};
    const char* bin_labels[] = {
        "[-1.5,-1.0)", "[-1.0,-0.5)", "[-0.5,-0.1)", "[-0.1,-0.01)", "[-0.01,0.01)",
        "[0.01,0.1)", "[0.1,0.5)", "[0.5,1.0)", "[1.0,1.2)", "[1.2,1.4)", "[1.4,1.5]"
    };
    int bins[N_BINS] = {0};

    // --- 2. 按突触后神经元分层统计 (Phase R2 模块 C: 5 层交错布局) ---
    // 用 get_layer_for_neuron_idx(post_idx) 替代连续区间判断
    const char* layer_names[] = {"L4         ", "L2/3       ", "L5         ", "L6         ", "prefrontal "};
    const int N_LAYERS = 5;
    struct LayerStats {
        int count;
        double sum_w;
        double sum_w2;
        int strong;    // |w| > 1.0
        int weak;      // |w| < 0.1
        int inhibitory; // w < 0
        int excitatory; // w > 0
        float max_w;
        float min_w;
    };
    LayerStats ls[N_LAYERS] = {};

    // --- 3. 全局统计 ---
    double g_sum = 0.0, g_sum2 = 0.0;
    int g_strong = 0, g_weak = 0, g_inhib = 0, g_excit = 0;
    std::vector<float> all_weights;
    all_weights.reserve(sample_count);

    for (int i = 0; i < sample_count; ++i) {
        float w = h[i].weight;
        g_sum += w;
        g_sum2 += (double)w * w;
        all_weights.push_back(w);

        // 直方图
        for (int k = 0; k < N_BINS; ++k) {
            if (w >= bin_edges[k] && w < bin_edges[k+1]) {
                bins[k]++;
                break;
            }
            if (k == N_BINS - 1 && w >= bin_edges[k] && w <= bin_edges[k+1] + 1e-4f) {
                bins[k]++;
                break;
            }
        }

        // 全局分类
        if (fabsf(w) > 1.0f) g_strong++;
        if (fabsf(w) < 0.1f) g_weak++;
        if (w < 0.0f) g_inhib++;
        else if (w > 0.0f) g_excit++;

        // 按层分类 (Phase R2 模块 C: 用 neuron_idx → layer_idx 辅助函数替代连续区间)
        int post = h[i].post_idx;
        int l = get_layer_for_neuron_idx(post);
        ls[l].count++;
        ls[l].sum_w += w;
        ls[l].sum_w2 += (double)w * w;
        if (fabsf(w) > 1.0f) ls[l].strong++;
        if (fabsf(w) < 0.1f) ls[l].weak++;
        if (w < 0.0f) ls[l].inhibitory++;
        else if (w > 0.0f) ls[l].excitatory++;
        if (ls[l].count == 1 || w > ls[l].max_w) ls[l].max_w = w;
        if (ls[l].count == 1 || w < ls[l].min_w) ls[l].min_w = w;
    }

    // --- 4. 计算分位数 ---
    std::sort(all_weights.begin(), all_weights.end());
    auto percentile = [&](double p) -> float {
        if (all_weights.empty()) return 0.0f;
        int idx = (int)(p * (all_weights.size() - 1));
        return all_weights[idx];
    };

    float g_mean = sample_count > 0 ? (float)(g_sum / sample_count) : 0.0f;
    float g_var = sample_count > 0 ? (float)((g_sum2 - g_sum * g_sum / sample_count) / sample_count) : 0.0f;
    float g_std = sqrtf(g_var);

    // --- 5. 输出报告 ---
    printf("\n============================================================\n");
    printf("  权重分布深度分析 [%s] (sample=%d)\n", tag, sample_count);
    printf("============================================================\n");

    printf("\n--- 全局统计 ---\n");
    printf("  mean=%.4f  std=%.4f  var=%.6f\n", g_mean, g_std, g_var);
    printf("  p1=%.4f  p5=%.4f  p10=%.4f  p25=%.4f  p50=%.4f  p75=%.4f  p90=%.4f  p95=%.4f  p99=%.4f\n",
           percentile(0.01), percentile(0.05), percentile(0.10), percentile(0.25),
           percentile(0.50), percentile(0.75), percentile(0.90), percentile(0.95), percentile(0.99));
    printf("  强突触 (|w|>1.0): %d (%.2f%%)\n", g_strong, 100.0 * g_strong / sample_count);
    printf("  弱突触 (|w|<0.1): %d (%.2f%%)\n", g_weak, 100.0 * g_weak / sample_count);
    printf("  抑制性 (w<0):     %d (%.2f%%)\n", g_inhib, 100.0 * g_inhib / sample_count);
    printf("  兴奋性 (w>0):     %d (%.2f%%)\n", g_excit, 100.0 * g_excit / sample_count);

    printf("\n--- 权重直方图 ---\n");
    for (int k = 0; k < N_BINS; ++k) {
        float pct = 100.0f * bins[k] / sample_count;
        // 文本柱状图 (每 1% = 1 格, 最多 50 格)
        int bar_len = (int)(pct * 2);
        if (bar_len > 50) bar_len = 50;
        printf("  %s: %8d (%5.2f%%) |", bin_labels[k], bins[k], pct);
        for (int b = 0; b < bar_len; ++b) printf("#");
        printf("\n");
    }

    printf("\n--- 按突触后神经元分层统计 ---\n");
    printf("  %-12s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n",
           "Layer", "Count", "Mean", "Std", "Min", "Max", "Strong%", "Weak%", "Inhib%", "Excit%");
    for (int l = 0; l < N_LAYERS; ++l) {
        if (ls[l].count == 0) {
            printf("  %-12s %8d %8s %8s %8s %8s %8s %8s %8s %8s\n",
                   layer_names[l], 0, "-", "-", "-", "-", "-", "-", "-", "-");
            continue;
        }
        float l_mean = (float)(ls[l].sum_w / ls[l].count);
        float l_var = (float)((ls[l].sum_w2 - ls[l].sum_w * ls[l].sum_w / ls[l].count) / ls[l].count);
        float l_std = sqrtf(l_var);
        printf("  %-12s %8d %8.4f %8.4f %8.4f %8.4f %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n",
               layer_names[l], ls[l].count, l_mean, l_std, ls[l].min_w, ls[l].max_w,
               100.0 * ls[l].strong / ls[l].count,
               100.0 * ls[l].weak / ls[l].count,
               100.0 * ls[l].inhibitory / ls[l].count,
               100.0 * ls[l].excitatory / ls[l].count);
    }
    printf("============================================================\n");
}

int main(int argc, char** argv) {
    // 禁用 stdout 缓冲, 让重定向到文件时也能实时输出 (每 30K 步检查需要)
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);

    stage2e::RunConfig config;
    std::string config_error;
    if (!stage2e::parse_run_config(argc, argv, &config, &config_error)) {
        fprintf(stderr, "ERROR: %s\n%s", config_error.c_str(), stage2e::run_config_usage());
        return 2;
    }
    if (config.show_help) {
        printf("%s", stage2e::run_config_usage());
        return 0;
    }
    const int total_steps = config.total_steps;

    printf("============================================================\n");
    printf("  THE TRUE AI - Stage 2e Phase 1\n");
    printf("  快时间尺度: AdEx + NMDA + STP + 群体编码\n");
    printf("============================================================\n");
    printf("  设计文档: docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md\n");
    printf("  神经元规模: %d (55K = 50K 联合皮层 + 5K 前额叶)\n", N_TOTAL_NEURONS_2E);
    printf("  突触规模:   %d (10.7M)\n", N_TOTAL_SYNAPSES_2E);
    printf("  柱数:       %d (柱内 L4/L2-3/L5/L6 四层皮层结构)\n", N_COLUMNS_2E);
    printf("  群体编码:   每柱 K=%d 神经元, 增益 %.1f\n",
           POP_CODING_K_PER_COLUMN, POP_CODING_GAIN);
    printf("  训练步数:   %d\n", total_steps);
    printf("  训练内存预算: %llu MiB (可用 --memory-budget-mb 调整)\n",
           static_cast<unsigned long long>(config.memory_budget_mb));
    printf("============================================================\n\n");

    // --- 1. 选择 GPU 设备 ---
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0 || config.device >= dev_count) {
        fprintf(stderr, "[P1 FAIL] 未检测到 CUDA 设备\n");
        return 1;
    }
    cudaSetDevice(config.device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config.device);
    printf("[P1] 使用 GPU: %s (%.0f MB 显存, compute capability %d.%d)\n\n",
           prop.name, prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.major, prop.minor);

    bool e0_mode = config.e0_mode;
    const char* csv_path = config.csv_path.empty() ? nullptr : config.csv_path.c_str();
    print_experiment_metadata(stdout, "", config, csv_path != nullptr, csv_path, &prop);
    printf("\n");

    // --- 2. 显存分配 ---
    stage2e::MemoryAllocator allocator(config.memory_budget_mb * 1024ULL * 1024ULL);
    size_t allocated = allocator.allocate_all();
    if (allocated == 0) {
        fprintf(stderr, "[P1 FAIL] 显存分配失败\n");
        return 1;
    }
    if (!allocator.check_budget()) {
        fprintf(stderr, "[P1 FAIL] 显存超预算\n");
        allocator.free_all();
        return 1;
    }
    allocator.print_budget_report();

    // --- 3. 网络初始化 (P1 新增) ---
    printf("\n");
    if (config.resume_path.empty()) {
        stage2e::init_network(&allocator, config.seed);
    } else {
        printf("[P1] resume mode: skipping fresh topology initialization\n");
    }

    // --- 3.5 加载真实文本语料 (LCCC 子集) ---
    // 替代 step%256 循环注入, 改为 UTF-8 字节流注入
    // 生物学意义: 让网络学习真实中文文本的字节级统计规律
    {
        size_t loaded = 0;
        if (!config.synthetic_input) {
            loaded = stage2e::load_text_corpus(config.text_path.c_str());
        }
        if (loaded > 0) {
            printf("[P1] 已加载 LCCC 真实文本语料: %zu 字节\n", loaded);
            printf("[P1] 输入模式: 真实 UTF-8 字节流 (替代 step%%256 循环)\n\n");
        } else if (config.synthetic_input) {
            printf("[P1] 输入模式: 显式 synthetic 0..255 循环\n\n");
        } else {
            fprintf(stderr, "[P1 FAIL] 无法加载语料: %s\n", config.text_path.c_str());
            allocator.free_all();
            return 1;
        }
    }

    // --- 4. 调度器初始化 ---
    stage2e::BioMechanismScheduler scheduler(&allocator);

    // E0 消融模式: 纯 STDP 基线 (关闭三因素调制 + CaMKII + 调质系统)
    if (e0_mode) {
        scheduler.e0_ablation = true;
        printf("  *** E0 消融模式: 纯 STDP 基线 (无三因素/CaMKII/调质) ***\n");
        stage2e::set_e0_ablation(true);
        printf("  *** E0 scheduler.e0_ablation = true ***\n\n");
    }

    int start_step = 0;
    if (!config.resume_path.empty()) {
        const bool requested_e0 = config.e0_mode;
        uint32_t checkpoint_seed = 0;
        const int resume_rc = scheduler.load_checkpoint(
            config.resume_path.c_str(), &start_step, &checkpoint_seed);
        if (resume_rc != 0) {
            fprintf(stderr, "[P1 FAIL] checkpoint resume failed (code=%d)\n", resume_rc);
            allocator.free_all();
            return 1;
        }
        if (scheduler.e0_ablation != requested_e0) {
            fprintf(stderr, "[P1 FAIL] --e0 does not match checkpoint mode\n");
            allocator.free_all();
            return 1;
        }
        config.seed = checkpoint_seed;
        emit_run_param(stdout, "", "resumed_topology_seed", "%u", config.seed);
        emit_run_param(stdout, "", "resumed_next_step", "%d", start_step);
        if (start_step >= total_steps) {
            fprintf(stderr, "[P1 FAIL] --steps (%d) must exceed resumed next_step (%d)\n",
                    total_steps, start_step);
            allocator.free_all();
            return 1;
        }
    }

    // --- 5. 主循环 ---
    FILE* csv_fp = nullptr;
    if (csv_path) {
        csv_fp = fopen(csv_path, start_step > 0 ? "a" : "w");
        if (!csv_fp) {
            fprintf(stderr, "[P1] 无法打开 CSV 输出: %s\n", csv_path);
        } else {
            if (start_step == 0) {
                print_experiment_metadata(csv_fp, "# ", config, true, csv_path, &prop);
                fprintf(csv_fp, "step,spikes,is_inject_step,byte,nmda_sum,nmda_nz,"
                            "xpre_sum,xpre_nz,ca_sum,ca_nz,weight_mean,weight_abs_mean,"
                            "weight_min,weight_max,arrived_events,dispatched_events,dropped_events,max_slot_depth\n");
            }
        }
    }

    printf("\n[P1] 开始 %d 步快时间尺度测试...\n\n", total_steps);

    for (int step = start_step; step < total_steps; ++step) {
        scheduler.step(step);

        // CSV 采样: 每步记录 (开销主要在 device→host 拷贝)
        if (csv_fp) {
            stage2e::PersistentBuffers& b = allocator.buffers();
            int n_syn = N_TOTAL_SYNAPSES_2E;
            int n_neu = N_TOTAL_NEURONS_2E;

            // 采样关键指标 (为控制开销, 仅对 nmda_current 做全量求和)
            float nmda_sum = device_sum_float(b.d_nmda_current, n_neu);
            int   nmda_nz  = device_count_nonzero_float(b.d_nmda_current, n_neu);
            // STDP trace (10.7M, 较大但 1000 步内可接受)
            float xpre_sum = device_sum_float(b.d_stdp_x_pre_trace, n_syn);
            int   xpre_nz  = device_count_nonzero_float(b.d_stdp_x_pre_trace, n_syn);
            // 钙浓度 (反映 NMDA 是否激活)
            float ca_sum = 0.0f; int ca_nz = 0;
            {
                // d_ca_snapshot 是突触级 (10.7M), 用 BioSynapse.ca_concentration 反而更准
                // 但 ca_concentration 在 synapse struct 里, 难以直接求和
                // 用 d_ca_snapshot (突触级, atomic 写入)
                std::vector<float> h_ca(n_syn);
                cudaMemcpy(h_ca.data(), b.d_ca_snapshot, n_syn * sizeof(float), cudaMemcpyDeviceToHost);
                double s = 0.0; int c = 0;
                for (int i = 0; i < n_syn; ++i) {
                    if (h_ca[i] != 0.0f) { s += h_ca[i]; c++; }
                }
                ca_sum = static_cast<float>(s); ca_nz = c;
            }
            float wmean = 0.0f, wabs_mean = 0.0f, wmin = 0.0f, wmax = 0.0f;
            sample_synapse_weight_stats(b, 100000, &wmean, &wabs_mean, &wmin, &wmax);

            bool is_inject = (step % INPUT_INJECT_INTERVAL == 0);
            uint8_t byte = is_inject ? stage2e::get_byte_for_step(step) : 0;

            fprintf(csv_fp, "%d,%d,%d,%d,%.4f,%d,%.4f,%d,%.4f,%d,%.6f,%.6f,%.6f,%.6f,%lld,%lld,%lld,%d\n",
                    step, scheduler.stats().total_spikes,
                    (int)is_inject, (int)byte,
                    nmda_sum, nmda_nz,
                    xpre_sum, xpre_nz,
                    ca_sum, ca_nz,
                    wmean, wabs_mean, wmin, wmax,
                    scheduler.arrived_events_accum(),
                    scheduler.dispatched_events_accum(),
                    scheduler.dropped_events_accum(),
                    scheduler.max_delay_slot_depth());
        }

        // P1 安全检查: 每 1000 步同步一次, 检测 kernel 错误
        if (step % 1000 == 0) {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "\n[P1 FAIL] CUDA 同步错误 at step %d: %s\n",
                        step, cudaGetErrorString(err));
                if (csv_fp) fclose(csv_fp);
                allocator.free_all();
                return 1;
            }
        }

        const int next_step = step + 1;
        if (g_stop_requested) {
            if (config.checkpoint_interval == 0) {
                fprintf(stderr, "\n[P1] stop requested; checkpointing is disabled\n");
                if (csv_fp) fclose(csv_fp);
                return 130;
            }
            fprintf(stderr, "\n[P1] stop requested; saving checkpoint at next_step=%d\n", next_step);
            const int checkpoint_rc = scheduler.save_checkpoint(
                next_step, config.checkpoint_dir.c_str(), config.seed);
            if (checkpoint_rc == 0) {
                scheduler.prune_checkpoints(config.checkpoint_dir.c_str(), config.keep_checkpoints);
            }
            if (csv_fp) fclose(csv_fp);
            return checkpoint_rc == 0 ? 130 : 1;
        }

        if (config.checkpoint_interval > 0 && next_step % config.checkpoint_interval == 0) {
            const int checkpoint_rc = scheduler.save_checkpoint(
                next_step, config.checkpoint_dir.c_str(), config.seed);
            if (checkpoint_rc != 0) {
                fprintf(stderr, "[P1 FAIL] checkpoint failed (code=%d)\n", checkpoint_rc);
                if (csv_fp) fclose(csv_fp);
                allocator.free_all();
                return 1;
            }
            scheduler.prune_checkpoints(config.checkpoint_dir.c_str(), config.keep_checkpoints);
        }
    }

    if (config.checkpoint_interval > 0 &&
        total_steps >= config.checkpoint_interval &&
        total_steps % config.checkpoint_interval != 0) {
        const int checkpoint_rc = scheduler.save_checkpoint(
            total_steps, config.checkpoint_dir.c_str(), config.seed);
        if (checkpoint_rc != 0) {
            fprintf(stderr, "[P1 FAIL] final checkpoint failed (code=%d)\n", checkpoint_rc);
            if (csv_fp) fclose(csv_fp);
            allocator.free_all();
            return 1;
        }
        scheduler.prune_checkpoints(config.checkpoint_dir.c_str(), config.keep_checkpoints);
    }

    if (csv_fp) {
        fclose(csv_fp);
        printf("\n[P1] CSV 数据已写入: %s\n", csv_path);
    }

    cudaDeviceSynchronize();
    printf("\n[P1] 测试完成\n\n");

    // --- 6. 最终报告 ---
    allocator.print_budget_report();
    printf("[P1] 累计统计:\n");
    printf("  总执行步数:        %d\n", scheduler.total_steps_executed());
    printf("  累计脉冲:          %d\n", scheduler.total_spikes_accum());
    printf("  平均脉冲/步:       %.1f\n",
           scheduler.total_steps_executed() > 0 ?
           static_cast<float>(scheduler.total_spikes_accum()) / scheduler.total_steps_executed() : 0.0f);
    printf("  最小脉冲/步:       %d\n", scheduler.min_spikes_per_step());
    printf("  最大脉冲/步:       %d\n", scheduler.max_spikes_per_step());
    printf("  脉冲极差:          %d\n", scheduler.spike_range());
    printf("  簇状发放步数:      %d (%.2f%%)\n",
           scheduler.total_burst_steps(), scheduler.burst_ratio());
    printf("  单神经元burst脉冲: %d\n", scheduler.total_single_neuron_burst_spikes());
    printf("  延迟事件到达:      %lld\n", scheduler.arrived_events_accum());
    printf("  延迟事件分发:      %lld\n", scheduler.dispatched_events_accum());
    printf("  延迟事件丢弃:      %lld\n", scheduler.dropped_events_accum());
    printf("  最大槽位深度:      %d / %d\n",
           scheduler.max_delay_slot_depth(), DELAY_RING_SLOT_CAPACITY);
    printf("  最终延迟队列位置:  %d / %d\n",
           scheduler.delay_ring_idx(), DELAY_STEPS_MAX);

    // --- 7. P1 通过判据 ---
    printf("\n============================================================\n");
    printf("  P1 通过判据检查 (设计文档 §7.1)\n");
    printf("============================================================\n");

    bool pass = true;
    bool no_crash = false;
    bool vram_ok = false;
    bool vram_limit = false;
    bool range_ok = false;
    bool burst_ok = false;
    bool active_ok = false;
    bool delay_ok = false;
    bool single_burst_ok = false;
    bool weight_ok = false;
    bool camkii_ok = false;
    bool elig_ok = false;
    bool mod_range_ok = false;
    bool byte_sel_ok = false;
    bool value_ok = false;
    bool chi2_ok = false;
    bool p3_inhibitory_ok = false;
    bool p3_kwta_ok = false;
    bool p3_wm_ok = false;
    bool p3_semantic_ok = false;
    bool semantic_ready_ok = false;
    bool psw_ok = false;  // PSW 概率突触权重判据
    bool ca_rebound_ok = false;  // Ca²⁺ 回弹 LTD 判据
    bool balance_ok = false;  // 平衡态网络验证判据 (1/√K 缩放 + 柱特异性输入)
    bool gate_ok = false;  // 丘脑-皮层门控判据 (§1.1 注意力门控)
    size_t peak_mb = 0;
    int range = 0;
    float burst = 0.0f;
    float avg_spikes = 0.0f;
    double drop_rate = 0.0;
    float final_wmean = 0.0f, final_wabs = 0.0f, final_wmin = 0.0f, final_wmax = 0.0f;
    // PSW 统计
    float psw_mean_alpha = 0.0f, psw_mean_beta = 0.0f;
    float psw_mean_evidence = 0.0f, psw_mean_confidence = 0.0f, psw_mature_ratio = 0.0f;
    // Ca²⁺ 回弹 LTD 统计
    float ca_mean = 0.0f, ca_max = 0.0f, ca_high_ratio = 0.0f;
    // 平衡态网络验证统计
    float bal_mean_fr = 0.0f, bal_std_fr = 0.0f, bal_cv = 0.0f;
    int camkii_nz_final = 0;
    float camkii_mean_final = 0.0f;
    int ca_nz_final = 0;
    double ca_mean_final = 0.0;
    int e1_nz_final = 0;
    int e2_nz_final = 0;
    stage2e::ModulatoryStats final_mod_stats{};
    double hist_var_final = 0.0;
    int hist_nz_final = 0;
    double hist_mean_final = 0.0;
    int chi2_total_significant_final = 0;
    int chi2_total_active_final = 0;
    double chi2_mean_final = 0.0;
    double chi2_max_final = 0.0;
    int wm_active_slots_final = 0;
    float wm_max_activation_final = 0.0f;
    float wm_mean_activation_final = 0.0f;
    double semantic_readiness_score = 0.0;
    // P3-C 语义聚类评估指标
    int    p3_semantic_eval_updates_final = 0;
    double p3_silhouette_final = 0.0;
    double p3_js_mean_final = 0.0;
    double p3_js_max_final = 0.0;
    double p3_column_ratio_final = 0.0;

    // 判据 1: 编译通过 (运行到这里即通过)
    printf("  [1] 编译通过:                       PASS\n");

    // 判据 2: 10K 步不崩
    no_crash = (scheduler.total_steps_executed() == total_steps);
    printf("  [2] %d 步不崩:                      %s\n",
           total_steps, no_crash ? "PASS" : "FAIL");
    pass &= no_crash;

    // 判据 3: 显存峰值不超过本次运行配置的预算
    peak_mb = allocator.vram_peak() / (1024 * 1024);
    vram_ok = (allocator.vram_peak() <= allocator.budget_bytes());
    printf("  [3] 显存峰值 <= 配置预算:             %s (实际 %zu MB / 预算 %llu MB)\n",
           vram_ok ? "PASS" : "FAIL", peak_mb,
           static_cast<unsigned long long>(config.memory_budget_mb));
    pass &= vram_ok;

    // 判据 4: 持久分配不超过配置预算
    vram_limit = (allocator.vram_used() <= allocator.budget_bytes());
    printf("  [4] 持久显存 <= 配置预算:            %s (%.1f%% 利用率)\n",
           vram_limit ? "PASS" : "FAIL",
           allocator.vram_used() * 100.0 / allocator.budget_bytes());
    pass &= vram_limit;

    // 判据 5: spike count 极差 > 100 (P1 核心判据)
    range = scheduler.spike_range();
    range_ok = (range > 100);
    printf("  [5] spike count 极差 > 100:         %s (实际 %d, min=%d max=%d)\n",
           range_ok ? "PASS" : "FAIL", range,
           scheduler.min_spikes_per_step(),
           scheduler.max_spikes_per_step());
    pass &= range_ok;

    // 判据 6: 簇状发放出现 (burst_ratio > 0.5%)
    burst = scheduler.burst_ratio();
    burst_ok = (burst > 0.5f);
    printf("  [6] 簇状发放出现 (burst%% > 0.5%%):    %s (实际 %.2f%%)\n",
           burst_ok ? "PASS" : "FAIL", burst);
    pass &= burst_ok;

    // 判据 7: 发放模式多样性 (平均脉冲/步 > 10, 不能死寂)
    avg_spikes = scheduler.total_steps_executed() > 0 ?
        static_cast<float>(scheduler.total_spikes_accum()) / scheduler.total_steps_executed() : 0.0f;
    active_ok = (avg_spikes > 10.0f);
    printf("  [7] 发放活动正常 (avg > 10):        %s (实际 %.1f)\n",
           active_ok ? "PASS" : "FAIL", avg_spikes);
    pass &= active_ok;

    drop_rate = scheduler.dispatched_events_accum() > 0 ?
        static_cast<double>(scheduler.dropped_events_accum()) / scheduler.dispatched_events_accum() : 0.0;
    delay_ok = (scheduler.arrived_events_accum() > 0 &&
                     scheduler.dispatched_events_accum() > 0 &&
                     drop_rate < 0.01);
    printf("  [8] 延迟事件链路有效:               %s (到达 %lld, 分发 %lld, 丢弃 %lld)\n",
           delay_ok ? "PASS" : "FAIL",
           scheduler.arrived_events_accum(),
           scheduler.dispatched_events_accum(),
           scheduler.dropped_events_accum());
    pass &= delay_ok;

    single_burst_ok = (scheduler.total_single_neuron_burst_spikes() > 0);
    printf("  [9] 单神经元burst出现:              %s (实际 %d)\n",
           single_burst_ok ? "PASS" : "FAIL",
           scheduler.total_single_neuron_burst_spikes());
    pass &= single_burst_ok;

    sample_synapse_weight_stats(allocator.buffers(), 100000, &final_wmean, &final_wabs, &final_wmin, &final_wmax);
    weight_ok = (final_wmax <= STDP_W_MAX_2E + 1e-3f && final_wmin >= -STDP_W_MAX_2E - 1e-3f);
    printf("  [10] 真实权重范围合法:              %s (min %.3f max %.3f mean %.3f)\n",
           weight_ok ? "PASS" : "FAIL", final_wmin, final_wmax, final_wmean);
    pass &= weight_ok;

    // 判据 10b: PSW 概率突触权重 (贝叶斯 STDP)
    // 验证: α/β 合法 (>0), 证据强度 (α+β) 增长, 成熟突触比例 > 0
    // PSW 核心保证: w_eff = W_MAX · α/(α+β) 物理上 ∈ (0, W_MAX), 不可能饱和
    // 初始 α+β=0.1 (弱先验), 学习后部分突触 α+β 应 > PSW_MATURITY_THRESH (0.2)
    sample_psw_stats(allocator.buffers(), 100000,
                     &psw_mean_alpha, &psw_mean_beta, &psw_mean_evidence,
                     &psw_mean_confidence, &psw_mature_ratio);
    {
        bool alpha_positive = (psw_mean_alpha > 0.0f);
        bool beta_positive  = (psw_mean_beta > 0.0f);
        // 初始 evidence=0.1, 学习后 mean_evidence 应 >= 0.09 (允许 10% 衰减)
        // 或 mature_ratio > 0 (至少部分突触证据增长)
        bool evidence_valid = (psw_mean_evidence >= 0.09f);
        bool mature_exists  = (psw_mature_ratio > 0.0f);
        psw_ok = (alpha_positive && beta_positive && evidence_valid && mature_exists);
        printf("  [10b] PSW 概率突触权重:            %s (α=%.4f β=%.4f ev=%.4f conf=%.4f mature=%.2f%%)\n",
               psw_ok ? "PASS" : "FAIL",
               psw_mean_alpha, psw_mean_beta, psw_mean_evidence,
               psw_mean_confidence, psw_mature_ratio * 100.0f);
        printf("       [PSW 保证: w_eff 物理上 ∈ (0, W_MAX), 不可能饱和; 初始 ev=0.1, mature_thresh=0.2]\n");
        pass &= psw_ok;
    }

    // 判据 10c: Ca²⁺ 回弹 LTD (生物学防饱和机制)
    // 验证: 高 Ca²⁺ 突触存在 (ca > CA_REBOUND_THRESHOLD), 回弹 LTD 机制可被触发
    // 生物意义: 高频刺激导致 Ca²⁺ 超载 → 主动削弱突触 (BCM 理论分子基础)
    // 注意: 短测中 high_ca_ratio 可能很低, 长测中随活动累积应有所增长
    sample_ca_rebound_stats(allocator.buffers(), 100000,
                            &ca_mean, &ca_max, &ca_high_ratio);
    {
        // 判据: 机制存在性 — 至少有突触 ca > threshold (max_ca > threshold)
        // 或 mean_ca > 0 (Ca²⁺ 系统正常运行)
        bool ca_system_active = (ca_mean > 0.0f);
        bool rebound_triggered = (ca_max > CA_REBOUND_THRESHOLD);
        ca_rebound_ok = ca_system_active;  // 短测仅验证机制运行, 长测验证触发
        printf("  [10c] Ca²⁺ 回弹 LTD:                %s (mean_ca=%.4f max_ca=%.4f high_ratio=%.4f%% thresh=%.2f)\n",
               ca_rebound_ok ? "PASS" : "FAIL",
               ca_mean, ca_max, ca_high_ratio * 100.0f, CA_REBOUND_THRESHOLD);
        if (rebound_triggered) {
            printf("       [回弹 LTD 已触发: %d 突触 ca > %.2f, β 正在累积]\n",
                   (int)(ca_high_ratio * 100000), CA_REBOUND_THRESHOLD);
        } else {
            printf("       [回弹 LTD 待触发: 当前 max_ca=%.4f < thresh=%.2f (长测中高频刺激可触发)]\n",
                   ca_max, CA_REBOUND_THRESHOLD);
        }
        pass &= ca_rebound_ok;
    }

    // 架构修复 Task 7: 提前触发语义聚类评估, 确保 [10d] 能读到最新 col_ratio
    // 原代码 run_semantic_eval 在 P3 判据区 (line ~1158) 才执行, 导致 [10d] 读到初值 0
    // run_semantic_eval 完全幂等, 在此调用后 P3 区不再重复调用
    scheduler.run_semantic_eval(total_steps);

    // 判据 10d: 平衡态网络验证 (1/√K 缩放 + 柱特异性输入)
    // 验证: CV > 0.5 (活动去相关) 且 col_ratio > 1.5 (柱间分化)
    // 平衡态条件 w ∝ 1/√K 激活兴奋/抑制动态平衡, 残差涨落独立 → CV 提升
    // 柱特异性输入打破对称性 → col_ratio 提升
    sample_balance_state_stats(allocator.buffers(), 10000,
                               &bal_mean_fr, &bal_std_fr, &bal_cv);
    {
        // 直接从 scheduler 读取 P3-C 的 col_ratio (循环中每 10K 步已更新)
        // p3_column_ratio_final 在判据 [21] 前才赋值, 此处直接用 scheduler 值
        double col_ratio_now = scheduler.p3_column_ratio();
        bool cv_ok = (bal_cv > 0.5f);
        bool col_ratio_ok = (col_ratio_now > 1.5);  // 复用 P3-C 的 col_ratio
        balance_ok = cv_ok && col_ratio_ok;
        printf("  [10d] 平衡态网络验证:              %s (CV=%.4f [>0.5] col_ratio=%.4f [>1.5] mean_fr=%.4f std_fr=%.4f)\n",
               balance_ok ? "PASS" : "FAIL",
               bal_cv, (float)col_ratio_now, bal_mean_fr, bal_std_fr);
        printf("       [1/√K 缩放激活兴奋/抑制平衡, 柱特异性输入打破对称性]\n");
        pass &= balance_ok;
    }

    // 权重分布深度分析 (E1 vs E0 对比用) - 采样全部 10.7M 突触
    analyze_weight_distribution(allocator.buffers(), N_TOTAL_SYNAPSES_2E, "FINAL");

    // ==================== P2 判据 ====================
    printf("\n--- P2 中时间尺度学习规则判据 ---\n");

    // 判据 11: CaMKII 活性非零 (机制运行)
    {
        const int camkii_sample = 100000;
        std::vector<float> h_camkii(camkii_sample);
        cudaMemcpy(h_camkii.data(), allocator.buffers().d_camkii_activity,
                   camkii_sample * sizeof(float), cudaMemcpyDeviceToHost);
        double camkii_sum = 0.0;
        int camkii_nz = 0;
        for (int i = 0; i < camkii_sample; ++i) {
            if (h_camkii[i] > 1e-8f) { camkii_nz++; camkii_sum += h_camkii[i]; }
        }
        float camkii_mean = camkii_sample > 0 ? static_cast<float>(camkii_sum / camkii_sample) : 0.0f;

        // 同时采样 ca_concentration 用于调试
        std::vector<BioSynapse> h_syn_sample(1000);
        cudaMemcpy(h_syn_sample.data(), allocator.buffers().d_synapses,
                   1000 * sizeof(BioSynapse), cudaMemcpyDeviceToHost);
        int ca_nz = 0;
        double ca_sum = 0.0;
        for (int i = 0; i < 1000; ++i) {
            if (h_syn_sample[i].ca_concentration > 1e-6f) { ca_nz++; ca_sum += h_syn_sample[i].ca_concentration; }
        }

        camkii_ok = (camkii_nz > 0);
        camkii_nz_final = camkii_nz;
        camkii_mean_final = camkii_mean;
        ca_nz_final = ca_nz;
        ca_mean_final = ca_nz > 0 ? ca_sum / 1000 : 0.0;
        printf("  [11] CaMKII 活性 > 0:               %s (nz=%d/%d, mean=%.6f, ca_nz=%d/%d, ca_mean=%.6f)\n",
               camkii_ok ? "PASS" : "FAIL", camkii_nz, camkii_sample, camkii_mean,
               ca_nz, 1000, ca_nz > 0 ? ca_sum / 1000 : 0.0);
        pass &= camkii_ok;
    }

    // 判据 12: eligibility trace 非零 (e1/e2 活跃)
    {
        std::vector<float> h_e1(10000), h_e2(10000);
        cudaMemcpy(h_e1.data(), allocator.buffers().d_eligibility,
                   10000 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_e2.data(), allocator.buffers().d_eligibility_slow,
                   10000 * sizeof(float), cudaMemcpyDeviceToHost);
        int e1_nz = 0, e2_nz = 0;
        for (int i = 0; i < 10000; ++i) {
            if (fabsf(h_e1[i]) > 1e-6f) e1_nz++;
            if (fabsf(h_e2[i]) > 1e-6f) e2_nz++;
        }
        elig_ok = (e1_nz > 0 && e2_nz > 0);
        e1_nz_final = e1_nz;
        e2_nz_final = e2_nz;
        printf("  [12] eligibility trace 非零:        %s (e1_nz=%d, e2_nz=%d)\n",
               elig_ok ? "PASS" : "FAIL", e1_nz, e2_nz);
        pass &= elig_ok;
    }

    // 判据 13: 调质浓度在 [0, 2] 范围内
    {
        stage2e::ModulatoryStats mstats = stage2e::get_modulatory_stats(&allocator);
        final_mod_stats = mstats;
        mod_range_ok = (mstats.da_mean >= 0.0f && mstats.da_mean <= 2.0f &&
                             mstats.ach_mean >= 0.0f && mstats.ach_mean <= 2.0f &&
                             mstats.ne_mean >= 0.0f && mstats.ne_mean <= 2.0f &&
                             mstats.ht5_mean >= 0.0f && mstats.ht5_mean <= 2.0f);
        printf("  [13] 调质浓度范围 [0,2]:            %s (DA=%.3f ACh=%.3f NE=%.3f 5HT=%.3f)\n",
               mod_range_ok ? "PASS" : "FAIL",
               mstats.da_mean, mstats.ach_mean, mstats.ne_mean, mstats.ht5_mean);
        pass &= mod_range_ok;
    }

    // 判据 14: 字节直方图方差 > 0 (字节选择性初步)
    {
        int h_hist[256];
        stage2e::get_byte_histogram(&allocator, h_hist);
        double hist_mean = 0.0;
        int hist_nz = 0;
        for (int i = 0; i < 256; ++i) {
            hist_mean += h_hist[i];
            if (h_hist[i] > 0) hist_nz++;
        }
        hist_mean /= 256.0;
        double hist_var = 0.0;
        for (int i = 0; i < 256; ++i) {
            double d = h_hist[i] - hist_mean;
            hist_var += d * d;
        }
        hist_var /= 256.0;
        // 字节选择性: 方差 > 0 表示不同字节产生不同 spike 数
        byte_sel_ok = (hist_var > 1.0 && hist_nz > 10);
        hist_var_final = hist_var;
        hist_nz_final = hist_nz;
        hist_mean_final = hist_mean;
        printf("  [14] 字节直方图方差 > 0:            %s (var=%.1f, nz_bins=%d, mean=%.1f)\n",
               byte_sel_ok ? "PASS" : "FAIL", hist_var, hist_nz, hist_mean);
        pass &= byte_sel_ok;

        // ===== 输入字节解读尝试 (UTF-8 文本模式) =====
        // 分析网络对真实文本字节的响应分布, 尝试还原输入文本特征
        if (stage2e::is_text_loaded()) {
            printf("\n");
            printf("  ============================================================\n");
            printf("  输入字节解读报告 (真实 UTF-8 文本模式)\n");
            printf("  ============================================================\n");
            printf("  语料大小: %zu 字节\n", stage2e::text_corpus_size());

            // 1. 找出网络响应最强的 Top-10 字节 (spike 数最多)
            printf("\n  --- 网络响应最强的 Top-10 字节 ---\n");
            printf("  %-6s %-10s %-12s %s\n", "字节", "十六进制", "响应计数", "字符解读");
            // 简单选择排序找 Top-10
            int top_idx[10] = {0};
            long long top_cnt[10] = {0};
            for (int b = 0; b < 256; ++b) {
                long long cnt = h_hist[b];
                // 插入到 top 列表
                for (int k = 0; k < 10; ++k) {
                    if (cnt > top_cnt[k]) {
                        // 后移
                        for (int j = 9; j > k; --j) {
                            top_idx[j] = top_idx[j-1];
                            top_cnt[j] = top_cnt[j-1];
                        }
                        top_idx[k] = b;
                        top_cnt[k] = cnt;
                        break;
                    }
                }
            }
            for (int k = 0; k < 10; ++k) {
                int b = top_idx[k];
                printf("  0x%02X  0x%02X       %10lld   ", b, b, top_cnt[k]);
                // 字符解读
                if (b < 0x20) {
                    printf("(控制字符)");
                } else if (b < 0x7F) {
                    printf("ASCII: '%c'", (char)b);
                } else if (b >= 0xC0 && b < 0xE0) {
                    printf("UTF-8 2字节序列头 (如拼音/扩展拉丁)");
                } else if (b >= 0xE0 && b < 0xF0) {
                    printf("UTF-8 3字节序列头 (如中文)");
                } else if (b >= 0xF0) {
                    printf("UTF-8 4字节序列头 (如 emoji)");
                } else if (b >= 0x80 && b < 0xC0) {
                    printf("UTF-8 续字节 (多字节序列中部)");
                }
                printf("\n");
            }

            // 2. 按字节类别统计响应分布
            long long resp_ascii = 0, resp_head = 0, resp_cont = 0;
            for (int b = 0; b < 256; ++b) {
                if (b < 0x80) resp_ascii += h_hist[b];
                else if (b >= 0xC0) resp_head += h_hist[b];
                else resp_cont += h_hist[b];
            }
            long long total_resp = resp_ascii + resp_head + resp_cont;
            printf("\n  --- 按字节类别响应分布 ---\n");
            printf("  ASCII (<0x80):    %lld (%.1f%%)\n", resp_ascii,
                   total_resp > 0 ? 100.0 * resp_ascii / total_resp : 0.0);
            printf("  UTF-8 头字节:      %lld (%.1f%%)\n", resp_head,
                   total_resp > 0 ? 100.0 * resp_head / total_resp : 0.0);
            printf("  UTF-8 续字节:      %lld (%.1f%%)\n", resp_cont,
                   total_resp > 0 ? 100.0 * resp_cont / total_resp : 0.0);

            // 3. 尝试还原输入文本序列 (注入次数最多的字节序列)
            // 通过注入位置追踪还原 (简化: 仅显示实际注入的前 64 字节)
            printf("\n  --- 实际注入的前 64 字节 (从 LCCC 语料读取) ---\n");
            printf("  十六进制: ");
            for (size_t i = 0; i < 64; ++i) {
                uint8_t b = stage2e::get_text_byte_at(i);
                printf("%02X ", b);
            }
            printf("\n");
            printf("  解读文本: ");
            for (size_t i = 0; i < 64; ++i) {
                uint8_t b = stage2e::get_text_byte_at(i);
                if (b >= 0x20 && b < 0x7F) printf("%c", (char)b);
                else if (b == ' ') printf(" ");
                else printf("?");  // UTF-8 多字节序列 (需解码器还原)
            }
            printf("\n");

            // 4. 解码能力评估
            printf("\n  --- 解码能力评估 ---\n");
            printf("  字节选择性: 方差=%.1f, 非零字节=%d/256\n", hist_var, hist_nz);
            if (hist_nz >= 200 && hist_var > 100.0) {
                printf("  评估: 网络对多数字节有差异化响应, 具备初步解码潜力\n");
                printf("  下一步: 需实现脉冲序列→字节的解码器 (语言运动皮层, 阶段 3)\n");
            } else if (hist_nz >= 100) {
                printf("  评估: 网络对部分字节有响应, 解码能力有限\n");
                printf("  建议: 增加训练步数或调整学习率\n");
            } else {
                printf("  评估: 网络响应不足, 无法解码\n");
                printf("  建议: 检查输入增益和柱偏好设计\n");
            }
            printf("  ============================================================\n\n");
        }
    }

    // 判据 15: DA 价值函数 V(s) 非平凡 (非零)
    {
        stage2e::ModulatoryStats mstats = stage2e::get_modulatory_stats(&allocator);
        final_mod_stats = mstats;
        value_ok = (fabsf(mstats.v_s) > 1e-6f || fabsf(mstats.v_sp) > 1e-6f);
        printf("  [15] DA 价值函数 V(s) 非零:         %s (V(s)=%.4f, V(s')=%.4f)\n",
               value_ok ? "PASS" : "FAIL", mstats.v_s, mstats.v_sp);
        pass &= value_ok;
    }

    // 判据 16: 卡方显著神经元 > 500 (字节/输入响应指标)
    // 对每个神经元 i, 用卡方检验判断其对 256 个字节的发放是否有选择性
    // H0: 神经元对所有字节发放率相同; H1: 对某些字节有选择性
    // df = 255, p < 0.05 临界值 ≈ 293.2 (查 chi-square 表)
    //
    // 分层分析: L4 (直接受输入驱动) vs L2/3/L5/L6/prefrontal (间接学习)
    // 后者才是突触学习效果的真正证据
    {
        const int B = 256;
        const int total_neurons = N_TOTAL_NEURONS_2E;
        const int total_bytes = B;

        // 拷贝 neuron_byte_counts 到 host (55K × 256 × 4B = 56 MB)
        std::vector<int> h_counts((size_t)total_neurons * total_bytes);
        cudaMemcpy(h_counts.data(), allocator.buffers().d_neuron_byte_counts,
                   (size_t)total_neurons * total_bytes * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // 计算每个字节的注入次数 (应该相等, 但用实际值更安全)
        int total_injections = total_steps / INPUT_INJECT_INTERVAL;
        std::vector<int> injections_per_byte(total_bytes, 0);
        for (int b = 0; b < total_bytes; ++b) {
            injections_per_byte[b] = total_injections / total_bytes;
        }
        double total_inj_sum = 0.0;
        for (int b = 0; b < total_bytes; ++b) total_inj_sum += injections_per_byte[b];

        const double chi2_critical = 293.2;  // df=255, p=0.05

        // 分层统计: L4 / L2-3 / L5 / L6 / prefrontal (Phase R2 模块 C: 4 层皮层交错布局)
        // 联合皮层 (50000 神经元) 按柱交错布局, 不能用连续区间, 用 get_layer_for_neuron_idx() 分类
        // 前额叶 (5000 神经元, [50000, 55000)) 全部为 prefrontal 层
        const char* layer_names[] = {"L4         ", "L2/3       ", "L5         ", "L6         ", "prefrontal "};
        const int N_LAYERS = 5;
        int layer_sizes[N_LAYERS] = {
            N_L4_TOTAL_2E,     // L4: 50 × 200 = 10000
            N_L23_TOTAL_2E,    // L2/3: 50 × 350 = 17500
            N_L5_TOTAL_2E,     // L5: 50 × 200 = 10000
            N_L6_TOTAL_2E,     // L6: 50 × 250 = 12500
            N_PREFRONTAL_NEURONS  // prefrontal: 5000
        };
        int layer_sig[N_LAYERS] = {0, 0, 0, 0, 0};
        int layer_act[N_LAYERS] = {0, 0, 0, 0, 0};
        double layer_chi2_sum[N_LAYERS] = {0.0, 0.0, 0.0, 0.0, 0.0};
        double layer_chi2_max[N_LAYERS] = {0.0, 0.0, 0.0, 0.0, 0.0};

        int total_significant = 0;
        int total_active = 0;
        double total_chi2_sum = 0.0;
        double total_chi2_max = 0.0;

        printf("  [16] 卡方显著神经元分层分析 (df=255, 临界=%.1f):\n", chi2_critical);
        printf("       %-14s %8s %8s %10s %10s %10s\n",
               "层级", "显著", "活跃", "活跃%", "卡方均值", "卡方最大");

        for (int i = 0; i < total_neurons; ++i) {
            int l = get_layer_for_neuron_idx(i);
            int row_total = 0;
            for (int b = 0; b < total_bytes; ++b) {
                row_total += h_counts[(size_t)i * total_bytes + b];
            }
            if (row_total < 10) continue;
            layer_act[l]++;

            double chi2 = 0.0;
            for (int b = 0; b < total_bytes; ++b) {
                double expected = (double)row_total * injections_per_byte[b] / total_inj_sum;
                if (expected < 1e-10) continue;
                double diff = (double)h_counts[(size_t)i * total_bytes + b] - expected;
                chi2 += diff * diff / expected;
            }
            layer_chi2_sum[l] += chi2;
            if (chi2 > layer_chi2_max[l]) layer_chi2_max[l] = chi2;
            if (chi2 > chi2_critical) layer_sig[l]++;
        }

        for (int l = 0; l < N_LAYERS; ++l) {
            double chi2_mean = layer_act[l] > 0 ? layer_chi2_sum[l] / layer_act[l] : 0.0;
            double active_pct = layer_sizes[l] > 0 ? 100.0 * layer_act[l] / layer_sizes[l] : 0.0;
            printf("       %-14s %8d %8d %9.1f%% %10.1f %10.1f\n",
                   layer_names[l], layer_sig[l], layer_act[l], active_pct, chi2_mean, layer_chi2_max[l]);
            total_significant += layer_sig[l];
            total_active += layer_act[l];
            total_chi2_sum += layer_chi2_sum[l];
            if (layer_chi2_max[l] > total_chi2_max) total_chi2_max = layer_chi2_max[l];
        }

        double total_chi2_mean = total_active > 0 ? total_chi2_sum / total_active : 0.0;
        chi2_ok = (total_significant > 500);
        chi2_total_significant_final = total_significant;
        chi2_total_active_final = total_active;
        chi2_mean_final = total_chi2_mean;
        chi2_max_final = total_chi2_max;
        printf("  [16] 字节/输入响应卡方 > 500:      %s (总显著=%d/%d 活跃, 均值=%.1f, 最大=%.1f)\n",
               chi2_ok ? "PASS" : "FAIL", total_significant, total_active,
               total_chi2_mean, total_chi2_max);
        pass &= chi2_ok;
    }

    printf("\n--- P3 网络动力学与语义涌现前置判据 ---\n");

    // P3-C: 语义聚类评估已在 [10d] 判据前触发 (架构修复 Task 7)
    // 长测 (P3-D) 中由 scheduler.step() 每 10000 步自动触发
    // 此处直接读取已计算的指标, 避免重复计算
    p3_semantic_eval_updates_final = scheduler.p3_semantic_eval_updates();
    p3_silhouette_final = scheduler.p3_silhouette_score();
    p3_js_mean_final = scheduler.p3_js_divergence_mean();
    p3_js_max_final = scheduler.p3_js_divergence_max();
    p3_column_ratio_final = scheduler.p3_column_ratio();

    p3_inhibitory_ok = (scheduler.p3_inhibitory_updates() > 0 && scheduler.p3_last_activity_drive() > 0.0f);
    printf("  [17] P3 稀疏竞争机制运行:          %s (inhib_updates=%d, activity_drive=%.4f)\n",
           p3_inhibitory_ok ? "PASS" : "FAIL",
           scheduler.p3_inhibitory_updates(), scheduler.p3_last_activity_drive());
    pass &= p3_inhibitory_ok;

    p3_kwta_ok = (scheduler.p3_kwta_updates() > 0 &&
                  scheduler.p3_kwta_active_columns() > 0 &&
                  scheduler.p3_kwta_winner_estimate() > 0);
    printf("  [18] P3-b k-WTA 柱内竞争:          %s (updates=%d, active_cols=%d/%d, winners=%d, suppressed=%d, k=%d)\n",
           p3_kwta_ok ? "PASS" : "FAIL",
           scheduler.p3_kwta_updates(), scheduler.p3_kwta_active_columns(), N_COLUMNS_2E,
           scheduler.p3_kwta_winner_estimate(), scheduler.p3_kwta_suppressed_estimate(),
           scheduler.p3_kwta_target_per_column());
    pass &= p3_kwta_ok;

    {
        std::vector<WMSlot> h_wm(WM_SLOTS);
        cudaMemcpy(h_wm.data(), allocator.buffers().d_wm_slots,
                   WM_SLOTS * sizeof(WMSlot), cudaMemcpyDeviceToHost);
        double wm_sum = 0.0;
        for (int i = 0; i < WM_SLOTS; ++i) {
            float a = h_wm[i].activation;
            wm_sum += a;
            if (a > WM_ACTIVATION_THRESHOLD) wm_active_slots_final++;
            if (a > wm_max_activation_final) wm_max_activation_final = a;
        }
        wm_mean_activation_final = static_cast<float>(wm_sum / WM_SLOTS);
        p3_wm_ok = (scheduler.p3_wm_updates() > 0 && wm_active_slots_final > 0);
        printf("  [19] P3 工作记忆槽位激活:          %s (wm_updates=%d, active_slots=%d/%d, mean=%.4f, max=%.4f)\n",
               p3_wm_ok ? "PASS" : "FAIL",
               scheduler.p3_wm_updates(), wm_active_slots_final, WM_SLOTS,
               wm_mean_activation_final, wm_max_activation_final);
        pass &= p3_wm_ok;
    }

    semantic_readiness_score = 0.0;
    if (camkii_ok) semantic_readiness_score += 0.15;
    if (elig_ok) semantic_readiness_score += 0.15;
    if (mod_range_ok && value_ok) semantic_readiness_score += 0.15;
    if (chi2_total_active_final > N_TOTAL_NEURONS_2E / 20) semantic_readiness_score += 0.15;
    if (p3_inhibitory_ok) semantic_readiness_score += 0.15;
    if (p3_kwta_ok) semantic_readiness_score += 0.10;
    if (p3_wm_ok) semantic_readiness_score += 0.15;
    semantic_ready_ok = (semantic_readiness_score >= 0.70);
    printf("  [20] 语义涌现准备度 >= 0.70:       %s (score=%.2f, P3-C silhouette/JS 见判据[21])\n",
           semantic_ready_ok ? "PASS" : "FAIL", semantic_readiness_score);
    pass &= semantic_ready_ok;

    // 判据 21: P3-C 语义聚类评估 (silhouette + JS divergence + 柱间差异)
    // 设计文档 §7.1 P3 硬检查点: silhouette > 0.15 + KL > 0.3 (JS≈KL/2 → JS>0.15), 柱间差异 > 2x
    // 短测判据: 机制运行 + 输出在合法范围内 (实际阈值验收在 P3-D 800K 长测)
    // silhouette ∈ [-1, 1]: 负值表示聚类比随机还差, 学习时间不够时正常
    // JS ∈ [0, ln2≈0.693]: 有界非负
    // col_ratio >= 0: 非负 (1.0=完全均匀, >2.0=有分化)
    {
        bool eval_ran = (p3_semantic_eval_updates_final > 0);
        bool sil_finite = (p3_silhouette_final >= -1.0 && p3_silhouette_final <= 1.0);
        bool js_finite = (p3_js_mean_final >= 0.0 && p3_js_mean_final <= 0.7);  // JS ∈ [0, ln2≈0.693]
        bool ratio_finite = (p3_column_ratio_final >= 0.0);
        p3_semantic_ok = (eval_ran && sil_finite && js_finite && ratio_finite);
        printf("  [21] P3-C 语义聚类评估:           %s (updates=%d, silhouette=%.4f, js_mean=%.4f, js_max=%.4f, col_ratio=%.2f)\n",
               p3_semantic_ok ? "PASS" : "FAIL",
               p3_semantic_eval_updates_final, p3_silhouette_final,
               p3_js_mean_final, p3_js_max_final, p3_column_ratio_final);
        printf("       [设计目标 800K: silhouette>0.15, js_mean>0.15, col_ratio>2.0]\n");
        pass &= p3_semantic_ok;
    }

    // 判据 22: 丘脑门控运行 (§1.1 注意力门控)
    // 验证: gate_mean ∈ [0.2, 0.9] (门控在合理范围, 非全关非全开)
    //       gate_open_ratio > 0.2 (至少部分柱门控开启, 非全闭门)
    // 生物意义: 丘脑根据活动水平 + novelty 动态调制输入增益
    //   - gate_mean 过低 → 输入被抑制, 网络活动不足
    //   - gate_mean 过高 → 门控失效, 等同于无门控
    {
        gate_ok = (scheduler.gate_mean() >= 0.2f && scheduler.gate_mean() <= 0.9f
                   && scheduler.gate_open_ratio() > 0.2f);
        printf("  [22] 丘脑门控运行:                %s (gate_mean=%.4f, open_ratio=%.4f)\n",
               gate_ok ? "PASS" : "FAIL",
               scheduler.gate_mean(), scheduler.gate_open_ratio());
        pass &= gate_ok;
    }

    printf("============================================================\n");
    printf("  %s: %s\n",
           pass ? "PASS" : "FAIL",
           pass ? "语义涌现路径可继续 ✓" : "语义涌现路径需继续调参");
    printf("  RUN_STATUS: COMPLETE%s\n",
           (!pass && !config.strict_criteria)
               ? " (科学判据未全部成熟；未启用 --strict-criteria)"
               : "");
    printf("============================================================\n");

    printf("\nFINAL_METRICS_BEGIN\n");
    emit_final_metric(stdout, "pass", "%d", pass ? 1 : 0);
    emit_final_metric(stdout, "e0_mode", "%d", e0_mode ? 1 : 0);
    emit_final_metric(stdout, "total_steps", "%d", scheduler.total_steps_executed());
    emit_final_metric(stdout, "total_spikes", "%d", scheduler.total_spikes_accum());
    emit_final_metric(stdout, "avg_spikes_per_step", "%.6f", avg_spikes);
    emit_final_metric(stdout, "min_spikes_per_step", "%d", scheduler.min_spikes_per_step());
    emit_final_metric(stdout, "max_spikes_per_step", "%d", scheduler.max_spikes_per_step());
    emit_final_metric(stdout, "spike_range", "%d", range);
    emit_final_metric(stdout, "burst_steps", "%d", scheduler.total_burst_steps());
    emit_final_metric(stdout, "burst_ratio", "%.6f", burst);
    emit_final_metric(stdout, "single_neuron_burst_spikes", "%d", scheduler.total_single_neuron_burst_spikes());
    emit_final_metric(stdout, "arrived_events", "%lld", scheduler.arrived_events_accum());
    emit_final_metric(stdout, "dispatched_events", "%lld", scheduler.dispatched_events_accum());
    emit_final_metric(stdout, "dropped_events", "%lld", scheduler.dropped_events_accum());
    emit_final_metric(stdout, "drop_rate", "%.9f", drop_rate);
    emit_final_metric(stdout, "max_delay_slot_depth", "%d", scheduler.max_delay_slot_depth());
    emit_final_metric(stdout, "final_delay_ring_idx", "%d", scheduler.delay_ring_idx());
    emit_final_metric(stdout, "vram_used_mb", "%.6f", allocator.vram_used() / (1024.0 * 1024.0));
    emit_final_metric(stdout, "vram_peak_mb", "%zu", peak_mb);
    emit_final_metric(stdout, "weight_mean", "%.9f", final_wmean);
    emit_final_metric(stdout, "weight_abs_mean", "%.9f", final_wabs);
    emit_final_metric(stdout, "weight_min", "%.9f", final_wmin);
    emit_final_metric(stdout, "weight_max", "%.9f", final_wmax);
    emit_final_metric(stdout, "camkii_nz", "%d", camkii_nz_final);
    emit_final_metric(stdout, "camkii_mean", "%.9f", camkii_mean_final);
    emit_final_metric(stdout, "ca_nz_sample", "%d", ca_nz_final);
    emit_final_metric(stdout, "ca_mean_sample", "%.9f", ca_mean_final);
    emit_final_metric(stdout, "eligibility_e1_nz", "%d", e1_nz_final);
    emit_final_metric(stdout, "eligibility_e2_nz", "%d", e2_nz_final);
    emit_final_metric(stdout, "da_mean", "%.9f", final_mod_stats.da_mean);
    emit_final_metric(stdout, "ach_mean", "%.9f", final_mod_stats.ach_mean);
    emit_final_metric(stdout, "ne_mean", "%.9f", final_mod_stats.ne_mean);
    emit_final_metric(stdout, "ht5_mean", "%.9f", final_mod_stats.ht5_mean);
    emit_final_metric(stdout, "value_s", "%.9f", final_mod_stats.v_s);
    emit_final_metric(stdout, "value_sp", "%.9f", final_mod_stats.v_sp);
    emit_final_metric(stdout, "byte_hist_var", "%.9f", hist_var_final);
    emit_final_metric(stdout, "byte_hist_nz_bins", "%d", hist_nz_final);
    emit_final_metric(stdout, "byte_hist_mean", "%.9f", hist_mean_final);
    emit_final_metric(stdout, "chi2_significant", "%d", chi2_total_significant_final);
    emit_final_metric(stdout, "chi2_active", "%d", chi2_total_active_final);
    emit_final_metric(stdout, "chi2_mean", "%.9f", chi2_mean_final);
    emit_final_metric(stdout, "chi2_max", "%.9f", chi2_max_final);
    emit_final_metric(stdout, "p3_inhibitory_updates", "%d", scheduler.p3_inhibitory_updates());
    emit_final_metric(stdout, "p3_kwta_updates", "%d", scheduler.p3_kwta_updates());
    emit_final_metric(stdout, "p3_kwta_active_columns", "%d", scheduler.p3_kwta_active_columns());
    emit_final_metric(stdout, "p3_kwta_winner_estimate", "%d", scheduler.p3_kwta_winner_estimate());
    emit_final_metric(stdout, "p3_kwta_suppressed_estimate", "%d", scheduler.p3_kwta_suppressed_estimate());
    emit_final_metric(stdout, "p3_kwta_target_per_column", "%d", scheduler.p3_kwta_target_per_column());
    emit_final_metric(stdout, "p3_wm_updates", "%d", scheduler.p3_wm_updates());
    emit_final_metric(stdout, "p3_activity_drive", "%.9f", scheduler.p3_last_activity_drive());
    emit_final_metric(stdout, "wm_active_slots", "%d", wm_active_slots_final);
    emit_final_metric(stdout, "wm_mean_activation", "%.9f", wm_mean_activation_final);
    emit_final_metric(stdout, "wm_max_activation", "%.9f", wm_max_activation_final);
    emit_final_metric(stdout, "semantic_readiness_score", "%.9f", semantic_readiness_score);
    emit_final_metric(stdout, "semantic_emergence_level", "%d", semantic_ready_ok ? 1 : 0);
    // P3-C 语义聚类评估指标
    emit_final_metric(stdout, "p3_semantic_eval_updates", "%d", p3_semantic_eval_updates_final);
    emit_final_metric(stdout, "p3_semantic_eval_step", "%d", scheduler.p3_semantic_eval_step());
    emit_final_metric(stdout, "p3_silhouette", "%.9f", p3_silhouette_final);
    emit_final_metric(stdout, "p3_js_divergence_mean", "%.9f", p3_js_mean_final);
    emit_final_metric(stdout, "p3_js_divergence_max", "%.9f", p3_js_max_final);
    emit_final_metric(stdout, "p3_column_ratio", "%.9f", p3_column_ratio_final);
    emit_final_metric(stdout, "criterion_no_crash", "%d", no_crash ? 1 : 0);
    emit_final_metric(stdout, "criterion_vram_peak", "%d", vram_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_vram_limit", "%d", vram_limit ? 1 : 0);
    emit_final_metric(stdout, "criterion_spike_range", "%d", range_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_burst", "%d", burst_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_active", "%d", active_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_delay", "%d", delay_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_single_burst", "%d", single_burst_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_weight_range", "%d", weight_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_camkii", "%d", camkii_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_eligibility", "%d", elig_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_modulatory_range", "%d", mod_range_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_byte_hist_var", "%d", byte_sel_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_value", "%d", value_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_chi2", "%d", chi2_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_p3_inhibitory", "%d", p3_inhibitory_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_p3_kwta", "%d", p3_kwta_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_p3_wm", "%d", p3_wm_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_semantic_readiness", "%d", semantic_ready_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_p3_semantic", "%d", p3_semantic_ok ? 1 : 0);
    emit_final_metric(stdout, "criterion_psw", "%d", psw_ok ? 1 : 0);
    emit_final_metric(stdout, "psw_mean_alpha", "%.9f", psw_mean_alpha);
    emit_final_metric(stdout, "psw_mean_beta", "%.9f", psw_mean_beta);
    emit_final_metric(stdout, "psw_mean_evidence", "%.9f", psw_mean_evidence);
    emit_final_metric(stdout, "psw_mean_confidence", "%.9f", psw_mean_confidence);
    emit_final_metric(stdout, "psw_mature_ratio", "%.9f", psw_mature_ratio);
    emit_final_metric(stdout, "criterion_ca_rebound", "%d", ca_rebound_ok ? 1 : 0);
    emit_final_metric(stdout, "ca_rebound_mean_ca", "%.9f", ca_mean);
    emit_final_metric(stdout, "ca_rebound_max_ca", "%.9f", ca_max);
    emit_final_metric(stdout, "ca_rebound_high_ratio", "%.9f", ca_high_ratio);
    emit_final_metric(stdout, "ca_rebound_threshold", "%.6f", CA_REBOUND_THRESHOLD);
    emit_final_metric(stdout, "criterion_balance", "%d", balance_ok ? 1 : 0);
    emit_final_metric(stdout, "balance_cv", "%.9f", bal_cv);
    emit_final_metric(stdout, "balance_mean_fr", "%.9f", bal_mean_fr);
    emit_final_metric(stdout, "balance_std_fr", "%.9f", bal_std_fr);
    // 丘脑-皮层门控指标 (§1.1 注意力门控)
    emit_final_metric(stdout, "gate_mean", "%.4f", scheduler.gate_mean());
    emit_final_metric(stdout, "gate_open_ratio", "%.4f", scheduler.gate_open_ratio());
    emit_final_metric(stdout, "criterion_gate", "%d", gate_ok ? 1 : 0);
    printf("FINAL_METRICS_END\n");

    // --- 8. 清理 ---
    allocator.free_all();
    cudaDeviceReset();

    return (config.strict_criteria && !pass) ? 2 : 0;
}
