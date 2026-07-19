// =============================================================================
// Stage 2e Phase 0: 基础框架烟雾测试
// =============================================================================
// 对应设计文档 §7.1 Phase 0:
//   - 基础框架搭建: BioSynapse 结构扩展、统一调度器、显存分配优化
//   - 通过条件: 编译通过, 10K 步不崩, 显存峰值 < 1332MB
//
// 用法:
//   snn_stage2e_p0                  # 默认 10K 步烟雾测试
//   snn_stage2e_p0 --steps 50000   # 自定义步数
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include "scheduler.cuh"
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
            printf("Usage: snn_stage2e_p0 [--steps N] [--help]\n");
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
    printf("  THE TRUE AI - Stage 2e Phase 0\n");
    printf("  多层级生物机制增强方案 v4 (基础框架烟雾测试)\n");
    printf("============================================================\n");
    printf("  设计文档: docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md\n");
    printf("  神经元规模: %d (55K = 50K 联合皮层 + 5K 前额叶)\n", N_TOTAL_NEURONS_2E);
    printf("  突触规模:   %d (10.7M)\n", N_TOTAL_SYNAPSES_2E);
    printf("  柱数:       %d\n", N_COLUMNS_2E);
    printf("  抑制亚型:   3 (FS/LTS/SOM)\n");
    printf("  训练步数:   %d\n", total_steps);
    printf("  显存预算:   1332 MB / 1500 MB (余量 168 MB)\n");
    printf("============================================================\n\n");

    // --- 1. 选择 GPU 设备 ---
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        fprintf(stderr, "[P0 FAIL] 未检测到 CUDA 设备\n");
        return 1;
    }
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[P0] 使用 GPU: %s (%.0f MB 显存, compute capability %d.%d)\n\n",
           prop.name, prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.major, prop.minor);

    // --- 2. 显存分配 ---
    stage2e::MemoryAllocator allocator;
    size_t allocated = allocator.allocate_all();
    if (allocated == 0) {
        fprintf(stderr, "[P0 FAIL] 显存分配失败\n");
        return 1;
    }
    if (!allocator.check_budget()) {
        fprintf(stderr, "[P0 FAIL] 显存超预算\n");
        allocator.free_all();
        return 1;
    }
    allocator.print_budget_report();

    // --- 3. 调度器初始化 ---
    stage2e::BioMechanismScheduler scheduler(&allocator);

    // --- 4. 主循环 ---
    printf("[P0] 开始 %d 步烟雾测试...\n\n", total_steps);

    for (int step = 0; step < total_steps; ++step) {
        scheduler.step(step);

        // P0 安全检查: 每 1000 步同步一次, 检测 kernel 错误
        if (step % 1000 == 0) {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "\n[P0 FAIL] CUDA 同步错误 at step %d: %s\n",
                        step, cudaGetErrorString(err));
                allocator.free_all();
                return 1;
            }
        }
    }

    cudaDeviceSynchronize();
    printf("\n[P0] 烟雾测试完成\n\n");

    // --- 5. 最终报告 ---
    allocator.print_budget_report();
    printf("[P0] 累计统计:\n");
    printf("  总执行步数:       %d\n", scheduler.total_steps_executed());
    printf("  累计占位脉冲:     %d\n", scheduler.total_spikes_accum());
    printf("  最终延迟队列位置: %d / %d\n", scheduler.delay_ring_idx(), DELAY_STEPS_MAX);

    // --- 6. P0 通过判据 ---
    printf("\n============================================================\n");
    printf("  P0 通过判据检查\n");
    printf("============================================================\n");

    bool pass = true;

    // 判据 1: 编译通过 (运行到这里即通过)
    printf("  [1] 编译通过:                    PASS\n");

    // 判据 2: 10K 步不崩
    bool no_crash = (scheduler.total_steps_executed() == total_steps);
    printf("  [2] %d 步不崩:                   %s\n",
           total_steps, no_crash ? "PASS" : "FAIL");
    pass &= no_crash;

    // 判据 3: 显存峰值 < 1332 MB
    size_t peak_mb = allocator.vram_peak() / (1024 * 1024);
    bool vram_ok = (peak_mb < 1332);
    printf("  [3] 显存峰值 < 1332 MB:          %s (实际 %zu MB)\n",
           vram_ok ? "PASS" : "FAIL", peak_mb);
    pass &= vram_ok;

    // 判据 4: 显存 < 1.5 GB 上限
    bool vram_limit = (allocator.vram_used() < VRAM_BUDGET_BYTES);
    printf("  [4] 显存 < 1.5GB 上限:           %s (%.1f%% 利用率)\n",
           vram_limit ? "PASS" : "FAIL",
           allocator.vram_used() * 100.0 / VRAM_BUDGET_BYTES);
    pass &= vram_limit;

    printf("============================================================\n");
    printf("  P0 总体: %s\n", pass ? "PASS ✓ 准备进入 Phase 1" : "FAIL ✗ 需修复");
    printf("============================================================\n");

    // --- 7. 清理 ---
    allocator.free_all();
    cudaDeviceReset();

    return pass ? 0 : 1;
}
