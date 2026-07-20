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
#include <cstdio>
#include <cstring>
#include <cstdlib>

// 解析命令行参数
static int parse_steps(int argc, char** argv) {
    int steps = SMOKE_TEST_STEPS_2E;  // 默认 10K
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[i + 1]);
            ++i;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: snn_stage2e_p1 [--steps N] [--help]\n");
            printf("  --steps N   运行 N 步 (默认 %d)\n", SMOKE_TEST_STEPS_2E);
            printf("  --help      显示帮助\n");
            exit(0);
        }
    }
    return steps;
}

int main(int argc, char** argv) {
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
    printf("\n[P1] 开始 %d 步快时间尺度测试...\n\n", total_steps);

    for (int step = 0; step < total_steps; ++step) {
        scheduler.step(step);

        // P1 安全检查: 每 1000 步同步一次, 检测 kernel 错误
        if (step % 1000 == 0) {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "\n[P1 FAIL] CUDA 同步错误 at step %d: %s\n",
                        step, cudaGetErrorString(err));
                allocator.free_all();
                return 1;
            }
        }
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

    printf("============================================================\n");
    printf("  P1 总体: %s\n", pass ? "PASS ✓ 准备进入 Phase 2" : "FAIL ✗ 需调参");
    printf("============================================================\n");

    // --- 8. 清理 ---
    allocator.free_all();
    cudaDeviceReset();

    return pass ? 0 : 1;
}
