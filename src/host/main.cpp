// =============================================================================
// main.cpp - SNN 阶段 0 MVP 入口
// =============================================================================
//
// 阶段 0 目标：验证 CUDA + SNN 训练管线跑通
//   - 神经元更新 ✓
//   - 突触传播 ✓
//   - STDP 学习 ✓
//   - 监控统计 ✓
//
// 成功判据：
//   1. 程序能编译运行
//   2. 显存占用符合预期（< 50 MB）
//   3. 训练过程中脉冲数、权重有变化
//   4. 无 CUDA 错误
// =============================================================================

#include "network.h"
#include "trainer.h"
#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "  SNN Dialogue - Stage 0 MVP" << std::endl;
    std::cout << "  纯 CUDA C++ + STDP 方案" << std::endl;
    std::cout << "============================================\n" << std::endl;

    // 检查 CUDA 设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "错误：未检测到 CUDA 设备" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[GPU] " << prop.name << std::endl;
    std::cout << "  显存:        " << prop.totalGlobalMem / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    std::cout << "  SM 数:       " << prop.multiProcessorCount << std::endl;
    std::cout << "  计算能力:    " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Warp size:   " << prop.warpSize << std::endl;
    std::cout << std::endl;

    // 创建网络
    SNNNetwork network(42);

    // 初始化
    network.initialize();

    // 训练配置（学习测试：加大学习强度 + 延长训练）
    TrainingConfig config;
    config.n_episodes = 500;           // 延长到 500 episode
    config.steps_per_episode = 500;
    config.dopamine_base = 1.0f;       // 基础多巴胺：中性学习
    config.dopamine_reward = 3.0f;     // 正向 reward：多巴胺 → 4.0（放大 STDP）
    config.dopamine_penalty = 0.5f;    // 负向 reward：多巴胺 → 0.5（抑制 STDP，但不为零）
    config.serotonin_base = 1.0f;
    config.log_interval = 25;          // 每 25 episode 打印一次
    config.save_interval = 100;
    config.checkpoint_dir = ".";
    config.enable_monitoring = true;

    // 模式识别任务配置
    config.n_patterns = 4;
    config.pattern_duration = 25;
    config.target_pattern = 0;
    config.motor_active_threshold = 0.015f;  // 15Hz 阈值

    // 创建训练器
    Trainer trainer(network, config);

    // 训练
    trainer.train();

    // 评估
    std::cout << "\n--- 评估 ---" << std::endl;
    float score = trainer.evaluate();
    std::cout << "评估分数: " << score << std::endl;

    // 清理
    network.cleanup();

    std::cout << "\n程序正常退出。" << std::endl;
    return 0;
}
