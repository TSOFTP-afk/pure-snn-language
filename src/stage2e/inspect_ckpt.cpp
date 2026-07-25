// 独立检查 v3 checkpoint 文件中 L6 神经元的字节响应计数
// 编译: g++ -O2 inspect_ckpt.cpp -o inspect_ckpt.exe
// 用法:
//   inspect_ckpt.exe <path-to-ckpt_stepNNNNN.snn2e>
//   inspect_ckpt.exe <path> --export-v4-buffers <output_dir>
//
// 支持新版 v3 格式 (scheduler_checkpoint.cu::save_checkpoint 写入)
// - 自动定位 "neuron_byte_counts" section (不依赖固定偏移)
// - 显示头部、section 表、L6 神经元响应统计
// - 兼容旧版 v2 格式时给出明确提示
//
// Task 20: --export-v4-buffers <dir> 导出 V4 新机制缓冲区统计到 CSV:
//   - pca_W_stats.csv      : PCA W 矩阵 (Frobenius 范数 + 每列 L2 范数 + 非零比例)
//   - hippo_stats.csv      : 海马索引 (填充率 + importance 分布 + replay_count 分布)
//   - coact_stats.csv      : 共激活 tracker (非零条目数 + coact_count 分布 + modulator_score 分布)
//   - wm_stats.csv         : WM 槽位 (激活槽位数 + activation 分布 + pattern 非零比例)
//   - w_pred_stats.csv     : W_pred 矩阵 (非对角项非零比例 + Frobenius 范数)

#include "ckpt_v3.h"
#include "config.h"
#include "types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <direct.h>   // _mkdir on Windows
#else
#  include <sys/stat.h> // mkdir on POSIX
#endif

using namespace stage2e;

// =============================================================================
// 辅助函数: 跨平台 mkdir
// =============================================================================
static int make_dir(const std::string& path) {
#if defined(_WIN32)
    return _mkdir(path.c_str());
#else
    return mkdir(path.c_str(), 0755);
#endif
}

// =============================================================================
// Task 20: V4 缓冲区导出实现
// =============================================================================

// 导出 PCA W 矩阵统计
//   布局: [N_TOTAL_NEURONS_2E × PATTERN_DIM] float (60K × 50 × 4B = 12MB)
//   注意: scheduler_checkpoint.cu 中 pca_W 维度是 N_TOTAL_NEURONS_2E × PATTERN_DIM
static bool export_pca_w_stats(CkptV3Reader& reader, const std::string& out_dir) {
    const auto* sec = reader.find_section("pca_W");
    if (!sec) {
        printf("  [跳过] pca_W section 未找到\n");
        return false;
    }
    const uint64_t expect = (uint64_t)N_TOTAL_NEURONS_2E * PATTERN_DIM * sizeof(float);
    printf("  pca_W: %llu bytes (期望 %llu)\n",
           (unsigned long long)sec->bytes, (unsigned long long)expect);
    if (sec->bytes != expect) {
        printf("  [警告] 字节数不匹配, 跳过\n");
        return false;
    }

    std::vector<float> W((size_t)N_TOTAL_NEURONS_2E * PATTERN_DIM, 0.0f);
    if (!reader.read_section_payload(*sec, W.data(), W.size() * sizeof(float))) {
        printf("  [错误] 读取 pca_W 失败\n");
        return false;
    }

    // 统计: Frobenius 范数 + 每列 L2 范数 + 非零比例
    double global_sq = 0.0;
    int    nonzero   = 0;
    std::vector<double> col_sq(PATTERN_DIM, 0.0);
    std::vector<int>    col_nonzero(PATTERN_DIM, 0);
    const int total_entries = N_TOTAL_NEURONS_2E * PATTERN_DIM;

    for (int n = 0; n < N_TOTAL_NEURONS_2E; ++n) {
        for (int k = 0; k < PATTERN_DIM; ++k) {
            float w = W[(size_t)n * PATTERN_DIM + k];
            double d = (double)w;
            global_sq += d * d;
            col_sq[k] += d * d;
            if (w != 0.0f) {
                nonzero++;
                col_nonzero[k]++;
            }
        }
    }
    double global_l2 = std::sqrt(global_sq);

    std::ofstream f(out_dir + "/pca_W_stats.csv");
    f << "metric,value\n";
    f << "frobenius_norm," << global_l2 << "\n";
    f << "total_entries," << total_entries << "\n";
    f << "nonzero_entries," << nonzero << "\n";
    f << "nonzero_ratio," << (double)nonzero / total_entries << "\n";
    f << "\nper_column_stats\n";
    f << "col_idx,l2_norm,nonzero_count,nonzero_ratio\n";
    for (int k = 0; k < PATTERN_DIM; ++k) {
        f << k << "," << std::sqrt(col_sq[k]) << ","
          << col_nonzero[k] << ","
          << (double)col_nonzero[k] / N_TOTAL_NEURONS_2E << "\n";
    }
    printf("  -> pca_W_stats.csv (Frobenius=%.4f, nonzero=%d/%d)\n",
           global_l2, nonzero, total_entries);
    return true;
}

// 导出海马索引统计
//   布局: [HIPP_INDEX_SIZE × HippoIndex] (50K × 256B = 12.8MB)
static bool export_hippo_stats(CkptV3Reader& reader, const std::string& out_dir) {
    const auto* sec = reader.find_section("hippo_indices");
    if (!sec) {
        printf("  [跳过] hippo_indices section 未找到\n");
        return false;
    }
    const uint64_t expect = (uint64_t)HIPP_INDEX_SIZE * sizeof(HippoIndex);
    printf("  hippo_indices: %llu bytes (期望 %llu)\n",
           (unsigned long long)sec->bytes, (unsigned long long)expect);
    if (sec->bytes != expect) {
        printf("  [警告] 字节数不匹配, 跳过\n");
        return false;
    }

    std::vector<HippoIndex> idx(HIPP_INDEX_SIZE);
    if (!reader.read_section_payload(*sec, idx.data(), idx.size() * sizeof(HippoIndex))) {
        printf("  [错误] 读取 hippo_indices 失败\n");
        return false;
    }

    // 统计: 填充率 + importance 分布 + replay_count 分布
    int filled = 0;
    double imp_sum = 0, imp_sq = 0;
    float imp_min = 1e30f, imp_max = -1e30f;
    int rc_sum = 0, rc_max = 0;
    for (int i = 0; i < HIPP_INDEX_SIZE; ++i) {
        // 填充判定: pattern_signature[0] != 0 或 importance > 0 或 replay_count > 0
        bool is_filled = (idx[i].importance > 0.0f) || (idx[i].replay_count > 0);
        if (is_filled) {
            filled++;
            imp_sum += idx[i].importance;
            imp_sq  += (double)idx[i].importance * idx[i].importance;
            if (idx[i].importance < imp_min) imp_min = idx[i].importance;
            if (idx[i].importance > imp_max) imp_max = idx[i].importance;
            rc_sum += idx[i].replay_count;
            if (idx[i].replay_count > rc_max) rc_max = idx[i].replay_count;
        }
    }
    double imp_mean = filled > 0 ? imp_sum / filled : 0.0;
    double imp_var  = filled > 0 ? imp_sq / filled - imp_mean * imp_mean : 0.0;
    double imp_std  = imp_var > 0 ? std::sqrt(imp_var) : 0.0;

    std::ofstream f(out_dir + "/hippo_stats.csv");
    f << "metric,value\n";
    f << "index_size," << HIPP_INDEX_SIZE << "\n";
    f << "filled_count," << filled << "\n";
    f << "fill_ratio," << (double)filled / HIPP_INDEX_SIZE << "\n";
    f << "importance_min," << (filled > 0 ? imp_min : 0.0) << "\n";
    f << "importance_max," << (filled > 0 ? imp_max : 0.0) << "\n";
    f << "importance_mean," << imp_mean << "\n";
    f << "importance_std," << imp_std << "\n";
    f << "replay_count_total," << rc_sum << "\n";
    f << "replay_count_max," << rc_max << "\n";
    f << "replay_count_avg," << (filled > 0 ? (double)rc_sum / filled : 0.0) << "\n";
    printf("  -> hippo_stats.csv (filled=%d/%d, imp=[%.4f,%.4f] mean=%.4f)\n",
           filled, HIPP_INDEX_SIZE,
           filled > 0 ? imp_min : 0.0f,
           filled > 0 ? imp_max : 0.0f,
           imp_mean);
    return true;
}

// 导出共激活 tracker 统计
//   布局: [COACT_TRACKER_SIZE × CoactTracker] (500K × 16B = 8MB)
static bool export_coact_stats(CkptV3Reader& reader, const std::string& out_dir) {
    const auto* sec = reader.find_section("coact_trackers");
    if (!sec) {
        printf("  [跳过] coact_trackers section 未找到\n");
        return false;
    }
    const uint64_t expect = (uint64_t)COACT_TRACKER_SIZE * sizeof(CoactTracker);
    printf("  coact_trackers: %llu bytes (期望 %llu)\n",
           (unsigned long long)sec->bytes, (unsigned long long)expect);
    if (sec->bytes != expect) {
        printf("  [警告] 字节数不匹配, 跳过\n");
        return false;
    }

    std::vector<CoactTracker> trk(COACT_TRACKER_SIZE);
    if (!reader.read_section_payload(*sec, trk.data(), trk.size() * sizeof(CoactTracker))) {
        printf("  [错误] 读取 coact_trackers 失败\n");
        return false;
    }

    // 统计: 非零条目数 + coact_count 分布 + modulator_score 分布
    int    nonzero   = 0;
    double cc_sum    = 0, cc_sq = 0;
    int    cc_max    = 0;
    double ms_sum    = 0, ms_sq = 0;
    float  ms_min    = 1e30f, ms_max = -1e30f;
    for (int i = 0; i < COACT_TRACKER_SIZE; ++i) {
        bool is_nonzero = (trk[i].candidate_pre != 0 || trk[i].coact_count != 0);
        if (is_nonzero) {
            nonzero++;
            cc_sum += trk[i].coact_count;
            cc_sq  += (double)trk[i].coact_count * trk[i].coact_count;
            if (trk[i].coact_count > cc_max) cc_max = trk[i].coact_count;
            ms_sum += trk[i].modulator_score;
            ms_sq  += (double)trk[i].modulator_score * trk[i].modulator_score;
            if (trk[i].modulator_score < ms_min) ms_min = trk[i].modulator_score;
            if (trk[i].modulator_score > ms_max) ms_max = trk[i].modulator_score;
        }
    }
    double cc_mean = nonzero > 0 ? cc_sum / nonzero : 0.0;
    double cc_var  = nonzero > 0 ? cc_sq / nonzero - cc_mean * cc_mean : 0.0;
    double cc_std  = cc_var > 0 ? std::sqrt(cc_var) : 0.0;
    double ms_mean = nonzero > 0 ? ms_sum / nonzero : 0.0;
    double ms_var  = nonzero > 0 ? ms_sq / nonzero - ms_mean * ms_mean : 0.0;
    double ms_std  = ms_var > 0 ? std::sqrt(ms_var) : 0.0;

    std::ofstream f(out_dir + "/coact_stats.csv");
    f << "metric,value\n";
    f << "tracker_size," << COACT_TRACKER_SIZE << "\n";
    f << "nonzero_count," << nonzero << "\n";
    f << "nonzero_ratio," << (double)nonzero / COACT_TRACKER_SIZE << "\n";
    f << "coact_count_max," << cc_max << "\n";
    f << "coact_count_mean," << cc_mean << "\n";
    f << "coact_count_std," << cc_std << "\n";
    f << "modulator_score_min," << (nonzero > 0 ? ms_min : 0.0) << "\n";
    f << "modulator_score_max," << (nonzero > 0 ? ms_max : 0.0) << "\n";
    f << "modulator_score_mean," << ms_mean << "\n";
    f << "modulator_score_std," << ms_std << "\n";
    printf("  -> coact_stats.csv (nonzero=%d/%d, cc_max=%d, ms_mean=%.4f)\n",
           nonzero, COACT_TRACKER_SIZE, cc_max, ms_mean);
    return true;
}

// 导出 WM 槽位统计
//   布局: [WM_SLOTS × WMSlot] (50 × 216B = 10.8KB)
static bool export_wm_stats(CkptV3Reader& reader, const std::string& out_dir) {
    const auto* sec = reader.find_section("wm_slots");
    if (!sec) {
        printf("  [跳过] wm_slots section 未找到\n");
        return false;
    }
    const uint64_t expect = (uint64_t)WM_SLOTS * sizeof(WMSlot);
    printf("  wm_slots: %llu bytes (期望 %llu)\n",
           (unsigned long long)sec->bytes, (unsigned long long)expect);
    if (sec->bytes != expect) {
        printf("  [警告] 字节数不匹配, 跳过\n");
        return false;
    }

    std::vector<WMSlot> slots(WM_SLOTS);
    if (!reader.read_section_payload(*sec, slots.data(), slots.size() * sizeof(WMSlot))) {
        printf("  [错误] 读取 wm_slots 失败\n");
        return false;
    }

    // 统计: 激活槽位数 (activation > WM_INJECT_THRESHOLD=0.3) + activation 分布 + pattern 非零比例
    int    active = 0;
    int    filled = 0;
    double act_sum = 0, act_sq = 0;
    float  act_min = 1e30f, act_max = -1e30f;
    int    pattern_nonzero_total = 0;
    const int pattern_dim = WM_PATTERN_DIM;
    const int total_pattern_entries = WM_SLOTS * pattern_dim;
    for (int i = 0; i < WM_SLOTS; ++i) {
        bool is_filled = (slots[i].activation > 0.0f);
        if (is_filled) {
            filled++;
            act_sum += slots[i].activation;
            act_sq  += (double)slots[i].activation * slots[i].activation;
            if (slots[i].activation < act_min) act_min = slots[i].activation;
            if (slots[i].activation > act_max) act_max = slots[i].activation;
            if (slots[i].activation > WM_INJECT_THRESHOLD) active++;
            for (int k = 0; k < pattern_dim; ++k) {
                if (slots[i].pattern[k] != 0.0f) pattern_nonzero_total++;
            }
        }
    }
    double act_mean = filled > 0 ? act_sum / filled : 0.0;
    double act_var  = filled > 0 ? act_sq / filled - act_mean * act_mean : 0.0;
    double act_std  = act_var > 0 ? std::sqrt(act_var) : 0.0;

    std::ofstream f(out_dir + "/wm_stats.csv");
    f << "metric,value\n";
    f << "wm_slots," << WM_SLOTS << "\n";
    f << "filled_count," << filled << "\n";
    f << "active_count," << active << "\n";
    f << "active_threshold," << WM_INJECT_THRESHOLD << "\n";
    f << "activation_min," << (filled > 0 ? act_min : 0.0) << "\n";
    f << "activation_max," << (filled > 0 ? act_max : 0.0) << "\n";
    f << "activation_mean," << act_mean << "\n";
    f << "activation_std," << act_std << "\n";
    f << "pattern_nonzero_entries," << pattern_nonzero_total << "\n";
    f << "pattern_total_entries," << total_pattern_entries << "\n";
    f << "pattern_nonzero_ratio," << (double)pattern_nonzero_total / total_pattern_entries << "\n";
    printf("  -> wm_stats.csv (filled=%d/%d, active=%d, act=[%.4f,%.4f])\n",
           filled, WM_SLOTS, active,
           filled > 0 ? act_min : 0.0f,
           filled > 0 ? act_max : 0.0f);
    return true;
}

// 导出 W_pred 矩阵统计
//   布局: [W_PRED_DIM × W_PRED_DIM] float (200 × 200 × 4B = 160KB)
static bool export_w_pred_stats(CkptV3Reader& reader, const std::string& out_dir) {
    const auto* sec = reader.find_section("w_pred");
    if (!sec) {
        printf("  [跳过] w_pred section 未找到\n");
        return false;
    }
    const uint64_t expect = (uint64_t)W_PRED_DIM * W_PRED_DIM * sizeof(float);
    printf("  w_pred: %llu bytes (期望 %llu)\n",
           (unsigned long long)sec->bytes, (unsigned long long)expect);
    if (sec->bytes != expect) {
        printf("  [警告] 字节数不匹配, 跳过\n");
        return false;
    }

    std::vector<float> W((size_t)W_PRED_DIM * W_PRED_DIM, 0.0f);
    if (!reader.read_section_payload(*sec, W.data(), W.size() * sizeof(float))) {
        printf("  [错误] 读取 w_pred 失败\n");
        return false;
    }

    // 统计: Frobenius 范数 + 非对角项非零比例 + 对角项统计
    double global_sq = 0.0;
    int    diag_nonzero    = 0;
    int    offdiag_nonzero = 0;
    double diag_sq         = 0.0;
    double offdiag_sq      = 0.0;
    const int total_offdiag = W_PRED_DIM * (W_PRED_DIM - 1);
    float  max_abs = 0.0f;
    for (int j = 0; j < W_PRED_DIM; ++j) {
        for (int k = 0; k < W_PRED_DIM; ++k) {
            float w = W[(size_t)j * W_PRED_DIM + k];
            double d = (double)w;
            global_sq += d * d;
            float aw = w < 0 ? -w : w;
            if (aw > max_abs) max_abs = aw;
            if (j == k) {
                diag_sq += d * d;
                if (w != 0.0f) diag_nonzero++;
            } else {
                offdiag_sq += d * d;
                if (w != 0.0f) offdiag_nonzero++;
            }
        }
    }
    double global_l2  = std::sqrt(global_sq);
    double diag_l2    = std::sqrt(diag_sq);
    double offdiag_l2 = std::sqrt(offdiag_sq);

    std::ofstream f(out_dir + "/w_pred_stats.csv");
    f << "metric,value\n";
    f << "matrix_dim," << W_PRED_DIM << "\n";
    f << "total_entries," << (W_PRED_DIM * W_PRED_DIM) << "\n";
    f << "frobenius_norm," << global_l2 << "\n";
    f << "max_abs_entry," << max_abs << "\n";
    f << "diag_l2_norm," << diag_l2 << "\n";
    f << "diag_nonzero," << diag_nonzero << "\n";
    f << "diag_nonzero_ratio," << (double)diag_nonzero / W_PRED_DIM << "\n";
    f << "offdiag_l2_norm," << offdiag_l2 << "\n";
    f << "offdiag_nonzero," << offdiag_nonzero << "\n";
    f << "offdiag_nonzero_ratio," << (double)offdiag_nonzero / total_offdiag << "\n";
    printf("  -> w_pred_stats.csv (Frobenius=%.4f, offdiag_nonzero=%d/%d)\n",
           global_l2, offdiag_nonzero, total_offdiag);
    return true;
}

// 导出所有 V4 缓冲区统计
static int export_v4_buffers(CkptV3Reader& reader, const std::string& out_dir) {
    printf("\n=== 导出 V4 缓冲区统计到 %s ===\n", out_dir.c_str());
    // 创建输出目录 (若不存在)
    make_dir(out_dir);

    int success = 0;
    if (export_pca_w_stats(reader, out_dir))   success++;
    if (export_hippo_stats(reader, out_dir))   success++;
    if (export_coact_stats(reader, out_dir))   success++;
    if (export_wm_stats(reader, out_dir))      success++;
    if (export_w_pred_stats(reader, out_dir))  success++;

    printf("\n=== 导出完成: %d/5 个 CSV 文件 ===\n", success);
    return success == 5 ? 0 : 1;
}

// =============================================================================
// 主程序
// =============================================================================
int main(int argc, char** argv) {
    // 参数解析: inspect_ckpt.exe <ckpt_path> [--export-v4-buffers <dir>]
    if (argc < 2) {
        std::printf("用法:\n");
        std::printf("  %s <path-to-ckpt.snn2e>\n", argv[0]);
        std::printf("  %s <path> --export-v4-buffers <output_dir>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    // 解析可选 --export-v4-buffers
    std::string export_dir;
    bool do_export = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export-v4-buffers" && i + 1 < argc) {
            do_export = true;
            export_dir = argv[++i];
        }
    }

    printf("=== Stage 2e v3 Checkpoint Inspector ===\n");
    printf("文件: %s\n\n", path);

    CkptV3Reader reader;
    if (!reader.open(path)) {
        std::fprintf(stderr, "\n[提示] 若文件是旧版 v2 格式 (.bin, magic=0x53434B50),\n"
                             "请用旧版 inspect_ckpt_v2.cpp 查看, 或重新训练生成 v3 格式 (.snn2e)\n");
        return 1;
    }

    // 1. 头部信息
    const auto& hdr = reader.header();
    printf("--- 头部 ---\n");
    printf("  version:           %u\n", hdr.version);
    printf("  section_count:     %u\n", hdr.section_count);
    printf("  payload_bytes:     %llu (%.2f MiB)\n",
           (unsigned long long)hdr.payload_bytes,
           hdr.payload_bytes / (1024.0 * 1024.0));
    printf("  payload_checksum:  %016llx\n",
           (unsigned long long)hdr.payload_checksum);
    printf("  n_neurons:         %u\n", hdr.n_neurons);
    printf("  n_synapses:        %u\n", hdr.n_synapses);
    printf("  bio_synapse_bytes: %u\n", hdr.bio_synapse_bytes);
    printf("  neuron_state_bytes:%u\n", hdr.neuron_state_bytes);

    int next_step = 0;
    if (reader.read_next_step(&next_step)) {
        printf("  next_step:         %d\n", next_step);
        printf("  topology_seed:     %u\n", reader.topology_seed());
    }

    // 2. section 表 (前 10 个 + neuron_byte_counts)
    printf("\n--- Section 表 (%zu 项) ---\n", reader.sections().size());
    const int show_n = static_cast<int>(reader.sections().size()) < 10
                       ? static_cast<int>(reader.sections().size()) : 10;
    for (int i = 0; i < show_n; ++i) {
        const auto& s = reader.sections()[i];
        printf("  [%2d] %-30s  %10llu bytes  (offset %llu)\n",
               i, s.name.c_str(),
               (unsigned long long)s.bytes,
               (unsigned long long)s.file_offset);
    }
    if (reader.sections().size() > 10) {
        printf("  ... 还有 %zu 项 ...\n", reader.sections().size() - 10);
    }

    // 3. 定位 neuron_byte_counts section
    const auto* nbc = reader.find_section("neuron_byte_counts");
    if (!nbc) {
        std::fprintf(stderr, "\n[错误] 'neuron_byte_counts' section 未找到\n");
        return 2;
    }
    printf("\n--- neuron_byte_counts section ---\n");
    printf("  字节数: %llu\n", (unsigned long long)nbc->bytes);
    const uint64_t expect_bytes = (uint64_t)N_TOTAL_NEURONS_2E * 256 * sizeof(int);
    printf("  期望值: %llu (N_TOTAL_NEURONS_2E=%d × 256 × 4)\n",
           (unsigned long long)expect_bytes, N_TOTAL_NEURONS_2E);
    if (nbc->bytes != expect_bytes) {
        std::fprintf(stderr, "[警告] 字节数不匹配!\n");
    }

    // 4. 读入完整 neuron_byte_counts 矩阵 (55K × 256 × 4B = 56MB)
    std::vector<int> byte_counts(N_TOTAL_NEURONS_2E * 256, 0);
    if (!reader.read_section_payload(*nbc, byte_counts.data(),
                                     byte_counts.size() * sizeof(int))) {
        std::fprintf(stderr, "[错误] 读取 neuron_byte_counts 失败\n");
        return 3;
    }
    printf("  已加载 %zu 个 int (%.1f MiB)\n",
           byte_counts.size(),
           byte_counts.size() * sizeof(int) / (1024.0 * 1024.0));

    // 5. L6 神经元布局 (与 network_init.cu 一致)
    const int N_COL = N_COLUMNS_2E;            // 50
    const int PER_COL = NEURONS_PER_COLUMN_2E; // 1000
    const int L6_OFF = COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E; // 750
    const int L6_SIZE = COL_L6_SIZE_2E;        // 250

    auto row_total = [&](int global_idx) -> int {
        const int* row = &byte_counts[(size_t)global_idx * 256];
        int sum = 0;
        for (int b = 0; b < 256; ++b) sum += row[b];
        return sum;
    };

    // 6. 采样: col 0, 25, 49 的 L6
    printf("\n--- L6 采样统计 ---\n");
    int sample_cols[] = {0, 25, 49};
    for (int sc : sample_cols) {
        int n_active = 0;
        int total_count = 0;
        int max_row_total = 0;
        for (int i = 0; i < L6_SIZE; ++i) {
            int g_idx = sc * PER_COL + L6_OFF + i;
            int rt = row_total(g_idx);
            if (rt > 0) {
                n_active++;
                total_count += rt;
                if (rt > max_row_total) max_row_total = rt;
            }
        }
        printf("  col %2d L6: %d/%d active, 总计数=%d, 最大单行计数=%d\n",
               sc, n_active, L6_SIZE, total_count, max_row_total);
    }

    // 7. 全 L6 统计 (50 柱 × 250 = 12500)
    int all_active = 0;
    int all_total = 0;
    for (int c = 0; c < N_COL; ++c) {
        for (int i = 0; i < L6_SIZE; ++i) {
            int g_idx = c * PER_COL + L6_OFF + i;
            int rt = row_total(g_idx);
            if (rt > 0) {
                all_active++;
                all_total += rt;
            }
        }
    }
    printf("\n--- 全 L6 统计 ---\n");
    printf("  active: %d/%d (%.1f%%)\n",
           all_active, N_COL * L6_SIZE,
           100.0 * all_active / (N_COL * L6_SIZE));
    printf("  总计数: %d\n", all_total);

    // 8. 顺带检查 col 0 的其他层
    int l4_active = 0, l23_active = 0, l5_active = 0;
    for (int i = 0; i < COL_L4_SIZE_2E; ++i) {
        if (row_total(i) > 0) l4_active++;
    }
    for (int i = COL_L4_SIZE_2E; i < COL_L4_SIZE_2E + COL_L23_SIZE_2E; ++i) {
        if (row_total(i) > 0) l23_active++;
    }
    for (int i = COL_L4_SIZE_2E + COL_L23_SIZE_2E;
         i < COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E; ++i) {
        if (row_total(i) > 0) l5_active++;
    }
    printf("\n--- col 0 其他层 ---\n");
    printf("  L4:   %d/%d active\n", l4_active, COL_L4_SIZE_2E);
    printf("  L2/3: %d/%d active\n", l23_active, COL_L23_SIZE_2E);
    printf("  L5:   %d/%d active\n", l5_active, COL_L5_SIZE_2E);

    // 9. decode_weights section 统计
    const auto* dw = reader.find_section("decode_weights");
    printf("\n--- decode_weights section ---\n");
    if (!dw) {
        printf("  [警告] decode_weights section 未找到 (旧版 checkpoint?)\n");
    } else {
        printf("  字节数: %llu\n", (unsigned long long)dw->bytes);
        const uint64_t dw_expect = (uint64_t)N_TOTAL_NEURONS_2E * 256 * sizeof(float);
        printf("  期望值: %llu (N_TOTAL_NEURONS_2E=%d × 256 × 4)\n",
               (unsigned long long)dw_expect, N_TOTAL_NEURONS_2E);
        if (dw->bytes != dw_expect) {
            printf("  [警告] 字节数不匹配!\n");
        }
        std::vector<float> weights((size_t)N_TOTAL_NEURONS_2E * 256, 0.0f);
        if (!reader.read_section_payload(*dw, weights.data(),
                                         weights.size() * sizeof(float))) {
            printf("  [错误] 读取 decode_weights 失败\n");
        } else {
            struct RowStat { int neuron_idx; float l2; float max_w; int argmax_byte; };
            std::vector<RowStat> rows(N_TOTAL_NEURONS_2E);
            double global_sq_sum = 0.0;
            float  global_max = 0.0f;
            int    nonzero_rows = 0;
            for (int n = 0; n < N_TOTAL_NEURONS_2E; ++n) {
                const float* row = &weights[(size_t)n * 256];
                float  row_max = 0.0f;
                int    row_argmax = 0;
                double row_sq = 0.0;
                for (int bb = 0; bb < 256; ++bb) {
                    float w = row[bb];
                    float aw = w < 0 ? -w : w;
                    if (aw > row_max) { row_max = aw; row_argmax = bb; }
                    row_sq += (double)w * w;
                }
                float row_l2 = (float)std::sqrt(row_sq);
                rows[n] = {n, row_l2, row_max, row_argmax};
                global_sq_sum += row_sq;
                if (row_max > global_max) global_max = row_max;
                if (row_l2 > 0.0f) nonzero_rows++;
            }
            double global_l2 = std::sqrt(global_sq_sum);
            printf("  L2 范数: %.2f\n", global_l2);
            printf("  最大权重: %.4f\n", global_max);
            printf("  非零行数: %d/%d\n", nonzero_rows, N_TOTAL_NEURONS_2E);
            const int top_n = N_TOTAL_NEURONS_2E < 5 ? N_TOTAL_NEURONS_2E : 5;
            std::partial_sort(rows.begin(), rows.begin() + top_n, rows.end(),
                [](const RowStat& a, const RowStat& b) { return a.l2 > b.l2; });
            printf("  Top-5 神经元 (按行 L2 范数):\n");
            for (int i = 0; i < top_n; ++i) {
                printf("    neuron[%d] L2=%.4f, argmax_byte=0x%02X\n",
                       rows[i].neuron_idx, rows[i].l2,
                       (unsigned)rows[i].argmax_byte);
            }
        }
    }

    // ==================== Task 20: V4 缓冲区导出 ====================
    if (do_export) {
        int err = export_v4_buffers(reader, export_dir);
        if (err != 0) {
            printf("\n[警告] 部分 V4 缓冲区导出失败 (见上方日志)\n");
        }
    }

    return 0;
}
