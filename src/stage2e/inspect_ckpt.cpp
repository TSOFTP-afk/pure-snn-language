// 独立检查 v3 checkpoint 文件中 L6 神经元的字节响应计数
// 编译: g++ -O2 inspect_ckpt.cpp -o inspect_ckpt.exe
// 用法: inspect_ckpt.exe <path-to-ckpt_stepNNNNN.snn2e>
//
// 支持新版 v3 格式 (scheduler_checkpoint.cu::save_checkpoint 写入)
// - 自动定位 "neuron_byte_counts" section (不依赖固定偏移)
// - 显示头部、section 表、L6 神经元响应统计
// - 兼容旧版 v2 格式时给出明确提示

#include "ckpt_v3.h"
#include "config.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace stage2e;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
        : "src/stage2e/checkpoints/ckpt_step50000.snn2e";
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
    //    柱内: L4(0-199) + L2/3(200-549) + L5(550-749) + L6(750-999)
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

    // 8. 顺带检查 col 0 的其他层 (确认 buffer 整体不是全 0)
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

    return 0;
}
