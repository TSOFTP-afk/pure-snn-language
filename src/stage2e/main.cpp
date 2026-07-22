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
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

// 解析命令行参数
static int parse_steps(int argc, char** argv) {
    int steps = SMOKE_TEST_STEPS_2E;  // 默认 10K
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[i + 1]);
            ++i;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: snn_stage2e_p1 [--steps N] [--csv PATH] [--help]\n");
            printf("  --steps N   运行 N 步 (默认 %d)\n", SMOKE_TEST_STEPS_2E);
            printf("  --csv PATH  每步输出 spike 序列到 CSV (评估模式)\n");
            printf("  --help      显示帮助\n");
            exit(0);
        }
    }
    return steps;
}

static const char* get_csv_path(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return nullptr;
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

int main(int argc, char** argv) {
    // 禁用 stdout 缓冲, 让重定向到文件时也能实时输出 (每 30K 步检查需要)
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int total_steps = parse_steps(argc, argv);

    printf("============================================================\n");
    printf("  THE TRUE AI - Stage 2e Phase 1\n");
    printf("  快时间尺度: AdEx + NMDA + STP + 群体编码\n");
    printf("============================================================\n");
    printf("  设计文档: docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md\n");
    printf("  神经元规模: %d (55K = 50K 联合皮层 + 5K 前额叶)\n", N_TOTAL_NEURONS_2E);
    printf("  突触规模:   %d (10.7M)\n", N_TOTAL_SYNAPSES_2E);
    printf("  柱数:       %d (柱内 sensory/assoc/motor 三层)\n", N_COLUMNS_2E);
    printf("  群体编码:   每柱 K=%d 神经元, 增益 %.1f\n",
           POP_CODING_K_PER_COLUMN, POP_CODING_GAIN);
    printf("  训练步数:   %d\n", total_steps);
    printf("  显存预算:   1332 MB / 1500 MB (余量 168 MB)\n");
    printf("============================================================\n\n");

    // --- 1. 选择 GPU 设备 ---
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        fprintf(stderr, "[P1 FAIL] 未检测到 CUDA 设备\n");
        return 1;
    }
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[P1] 使用 GPU: %s (%.0f MB 显存, compute capability %d.%d)\n\n",
           prop.name, prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.major, prop.minor);

    // --- 2. 显存分配 ---
    stage2e::MemoryAllocator allocator;
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
    stage2e::init_network(&allocator);

    // --- 4. 调度器初始化 ---
    stage2e::BioMechanismScheduler scheduler(&allocator);

    // --- 5. 主循环 ---
    const char* csv_path = get_csv_path(argc, argv);
    FILE* csv_fp = nullptr;
    if (csv_path) {
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp) {
            fprintf(stderr, "[P1] 无法打开 CSV 输出: %s\n", csv_path);
        } else {
            fprintf(csv_fp, "step,spikes,is_inject_step,byte,nmda_sum,nmda_nz,"
                            "xpre_sum,xpre_nz,ca_sum,ca_nz,weight_mean,weight_abs_mean,"
                            "weight_min,weight_max,arrived_events,dispatched_events,dropped_events,max_slot_depth\n");
        }
    }

    printf("\n[P1] 开始 %d 步快时间尺度测试...\n\n", total_steps);

    for (int step = 0; step < total_steps; ++step) {
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

    // 判据 1: 编译通过 (运行到这里即通过)
    printf("  [1] 编译通过:                       PASS\n");

    // 判据 2: 10K 步不崩
    bool no_crash = (scheduler.total_steps_executed() == total_steps);
    printf("  [2] %d 步不崩:                      %s\n",
           total_steps, no_crash ? "PASS" : "FAIL");
    pass &= no_crash;

    // 判据 3: 显存峰值 < 1332 MB
    size_t peak_mb = allocator.vram_peak() / (1024 * 1024);
    bool vram_ok = (peak_mb < 1332);
    printf("  [3] 显存峰值 < 1332 MB:             %s (实际 %zu MB)\n",
           vram_ok ? "PASS" : "FAIL", peak_mb);
    pass &= vram_ok;

    // 判据 4: 显存 < 1.5 GB 上限
    bool vram_limit = (allocator.vram_used() < VRAM_BUDGET_BYTES);
    printf("  [4] 显存 < 1.5GB 上限:              %s (%.1f%% 利用率)\n",
           vram_limit ? "PASS" : "FAIL",
           allocator.vram_used() * 100.0 / VRAM_BUDGET_BYTES);
    pass &= vram_limit;

    // 判据 5: spike count 极差 > 100 (P1 核心判据)
    int range = scheduler.spike_range();
    bool range_ok = (range > 100);
    printf("  [5] spike count 极差 > 100:         %s (实际 %d, min=%d max=%d)\n",
           range_ok ? "PASS" : "FAIL", range,
           scheduler.min_spikes_per_step(),
           scheduler.max_spikes_per_step());
    pass &= range_ok;

    // 判据 6: 簇状发放出现 (burst_ratio > 0.5%)
    float burst = scheduler.burst_ratio();
    bool burst_ok = (burst > 0.5f);
    printf("  [6] 簇状发放出现 (burst%% > 0.5%%):    %s (实际 %.2f%%)\n",
           burst_ok ? "PASS" : "FAIL", burst);
    pass &= burst_ok;

    // 判据 7: 发放模式多样性 (平均脉冲/步 > 10, 不能死寂)
    float avg_spikes = scheduler.total_steps_executed() > 0 ?
        static_cast<float>(scheduler.total_spikes_accum()) / scheduler.total_steps_executed() : 0.0f;
    bool active_ok = (avg_spikes > 10.0f);
    printf("  [7] 发放活动正常 (avg > 10):        %s (实际 %.1f)\n",
           active_ok ? "PASS" : "FAIL", avg_spikes);
    pass &= active_ok;

    double drop_rate = scheduler.dispatched_events_accum() > 0 ?
        static_cast<double>(scheduler.dropped_events_accum()) / scheduler.dispatched_events_accum() : 0.0;
    bool delay_ok = (scheduler.arrived_events_accum() > 0 &&
                     scheduler.dispatched_events_accum() > 0 &&
                     drop_rate < 0.01);
    printf("  [8] 延迟事件链路有效:               %s (到达 %lld, 分发 %lld, 丢弃 %lld)\n",
           delay_ok ? "PASS" : "FAIL",
           scheduler.arrived_events_accum(),
           scheduler.dispatched_events_accum(),
           scheduler.dropped_events_accum());
    pass &= delay_ok;

    bool single_burst_ok = (scheduler.total_single_neuron_burst_spikes() > 0);
    printf("  [9] 单神经元burst出现:              %s (实际 %d)\n",
           single_burst_ok ? "PASS" : "FAIL",
           scheduler.total_single_neuron_burst_spikes());
    pass &= single_burst_ok;

    float final_wmean = 0.0f, final_wabs = 0.0f, final_wmin = 0.0f, final_wmax = 0.0f;
    sample_synapse_weight_stats(allocator.buffers(), 100000, &final_wmean, &final_wabs, &final_wmin, &final_wmax);
    bool weight_ok = (final_wmax <= STDP_W_MAX_2E + 1e-3f && final_wmin >= -STDP_W_MAX_2E - 1e-3f);
    printf("  [10] 真实权重范围合法:              %s (min %.3f max %.3f mean %.3f)\n",
           weight_ok ? "PASS" : "FAIL", final_wmin, final_wmax, final_wmean);
    pass &= weight_ok;

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

        bool camkii_ok = (camkii_nz > 0);
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
        bool elig_ok = (e1_nz > 0 && e2_nz > 0);
        printf("  [12] eligibility trace 非零:        %s (e1_nz=%d, e2_nz=%d)\n",
               elig_ok ? "PASS" : "FAIL", e1_nz, e2_nz);
        pass &= elig_ok;
    }

    // 判据 13: 调质浓度在 [0, 2] 范围内
    {
        stage2e::ModulatoryStats mstats = stage2e::get_modulatory_stats(&allocator);
        bool mod_range_ok = (mstats.da_mean >= 0.0f && mstats.da_mean <= 2.0f &&
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
        bool byte_sel_ok = (hist_var > 1.0 && hist_nz > 10);
        printf("  [14] 字节直方图方差 > 0:            %s (var=%.1f, nz_bins=%d, mean=%.1f)\n",
               byte_sel_ok ? "PASS" : "FAIL", hist_var, hist_nz, hist_mean);
        pass &= byte_sel_ok;
    }

    // 判据 15: DA 价值函数 V(s) 非平凡 (非零)
    {
        stage2e::ModulatoryStats mstats = stage2e::get_modulatory_stats(&allocator);
        bool value_ok = (fabsf(mstats.v_s) > 1e-6f || fabsf(mstats.v_sp) > 1e-6f);
        printf("  [15] DA 价值函数 V(s) 非零:         %s (V(s)=%.4f, V(s')=%.4f)\n",
               value_ok ? "PASS" : "FAIL", mstats.v_s, mstats.v_sp);
        pass &= value_ok;
    }

    // 判据 16: 卡方显著神经元 > 500 (P2 硬验收, 设计文档 §7.1)
    // 对每个神经元 i, 用卡方检验判断其对 256 个字节的发放是否有选择性
    // H0: 神经元对所有字节发放率相同; H1: 对某些字节有选择性
    // df = 255, p < 0.05 临界值 ≈ 293.2 (查 chi-square 表)
    {
        const int N = N_TOTAL_NEURONS_2E;
        const int B = 256;
        const int total_neurons = N;
        const int total_bytes = B;

        // 拷贝 neuron_byte_counts 到 host (55K × 256 × 4B = 56 MB)
        std::vector<int> h_counts((size_t)total_neurons * total_bytes);
        cudaMemcpy(h_counts.data(), allocator.buffers().d_neuron_byte_counts,
                   (size_t)total_neurons * total_bytes * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // 计算每个字节的注入次数 (应该相等, 但用实际值更安全)
        // 总注入次数 = total_steps / INPUT_INJECT_INTERVAL
        int total_injections = total_steps / INPUT_INJECT_INTERVAL;
        std::vector<int> injections_per_byte(total_bytes, 0);
        for (int b = 0; b < total_bytes; ++b) {
            injections_per_byte[b] = total_injections / total_bytes;
        }
        // 总注入次数 (用 injections_per_byte 之和)
        double total_inj_sum = 0.0;
        for (int b = 0; b < total_bytes; ++b) total_inj_sum += injections_per_byte[b];

        // 卡方检验 (df=255, p<0.05 临界值 ≈ 293.2)
        // χ²_i = Σ_b (N_ib - E_ib)² / E_ib
        // E_ib = (Σ_b N_ib) × (injections_b / total_inj_sum)
        const double chi2_critical = 293.2;  // df=255, p=0.05
        int significant_neurons = 0;
        int active_neurons = 0;  // 至少有 1 次 spike 的神经元
        double chi2_sum = 0.0;
        double chi2_max = 0.0;

        for (int i = 0; i < total_neurons; ++i) {
            // 该神经元所有字节的总 spike 数
            int row_total = 0;
            for (int b = 0; b < total_bytes; ++b) {
                row_total += h_counts[(size_t)i * total_bytes + b];
            }
            if (row_total < 10) continue;  // 太少不检验
            active_neurons++;

            // 计算卡方统计量
            double chi2 = 0.0;
            for (int b = 0; b < total_bytes; ++b) {
                double expected = (double)row_total * injections_per_byte[b] / total_inj_sum;
                if (expected < 1e-10) continue;
                double diff = (double)h_counts[(size_t)i * total_bytes + b] - expected;
                chi2 += diff * diff / expected;
            }

            chi2_sum += chi2;
            if (chi2 > chi2_max) chi2_max = chi2;
            if (chi2 > chi2_critical) significant_neurons++;
        }

        double chi2_mean = active_neurons > 0 ? chi2_sum / active_neurons : 0.0;
        bool chi2_ok = (significant_neurons > 500);
        printf("  [16] 卡方显著神经元 > 500:        %s (显著=%d/%d 活跃, 临界=%.1f, 均值=%.1f, 最大=%.1f)\n",
               chi2_ok ? "PASS" : "FAIL", significant_neurons, active_neurons,
               chi2_critical, chi2_mean, chi2_max);
        pass &= chi2_ok;
    }

    printf("============================================================\n");
    printf("  %s: %s\n",
           pass ? "PASS" : "FAIL",
           pass ? "P1+P2 通过 ✓" : "需调参");
    printf("============================================================\n");

    // --- 8. 清理 ---
    allocator.free_all();
    cudaDeviceReset();

    return pass ? 0 : 1;
}
