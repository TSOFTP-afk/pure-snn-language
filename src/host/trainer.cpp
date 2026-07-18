// =============================================================================
// trainer.cpp - 训练循环实现
//
// 任务：模式识别
//   - 4 个输入模式 A/B/C/D
//   - 目标：Motor 在模式 A 时高活动，其他模式时低活动
//   - Reward 根据匹配度计算，通过多巴胺调制 STDP
// =============================================================================

#include "trainer.h"
#include "neuron.cuh"
#include "synapse.cuh"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

Trainer::Trainer(SNNNetwork& net, TrainingConfig config)
    : net_(net), config_(config), best_reward_(0.0f), episode_count_(0) {}

Trainer::~Trainer() {}

// -----------------------------------------------------------------------------
// 生成模式输入：4 个模式各激活感觉皮层的 1/4
//   模式 A: [0,                  N/4)
//   模式 B: [N/4,                N/2)
//   模式 C: [N/2,                3N/4)
//   模式 D: [3N/4,               N)
// 每个模式持续 pattern_duration 步，期间该子集以泊松方式发放
// -----------------------------------------------------------------------------
void Trainer::generate_pattern_input(float* h_input, int n_sensory, int step) {
    int pattern = current_pattern(step);
    int seg_size = n_sensory / config_.n_patterns;
    int seg_start = pattern * seg_size;
    int seg_end = (pattern == config_.n_patterns - 1) ? n_sensory : seg_start + seg_size;

    // 清零
    for (int i = 0; i < n_sensory; i++) {
        h_input[i] = 0.0f;
    }

    // 当前模式的子集以 30% 概率发放（强度 1.5）
    // 其他子集保持静默
    for (int i = seg_start; i < seg_end; i++) {
        if (std::rand() % 100 < 30) {
            h_input[i] = 1.5f;
        }
    }
}

// -----------------------------------------------------------------------------
// 计算 reward
//   目标模式（A）时：Motor 高活动 → +reward，低活动 → -reward（缺失）
//   非目标模式时：  Motor 低活动 → +reward，高活动 → -reward（误报）
//   reward ∈ [-1, 1]
// -----------------------------------------------------------------------------
float Trainer::compute_reward(const float* motor_output, int n_motor, int step) {
    // 计算 Motor 平均活动
    float sum = 0.0f;
    for (int i = 0; i < n_motor; i++) {
        sum += motor_output[i];
    }
    float mean_activity = sum / n_motor;

    bool is_target = is_target_pattern(step);
    bool is_active = mean_activity > config_.motor_active_threshold;

    float reward;
    if (is_target) {
        // 目标模式：希望高活动
        reward = is_active ? 1.0f : -0.5f;  // 命中 +1，缺失 -0.5
    } else {
        // 非目标模式：希望低活动
        reward = is_active ? -1.0f : 0.3f;  // 误报 -1，正确抑制 +0.3
    }
    return reward;
}

// -----------------------------------------------------------------------------
// 根据 reward 设置多巴胺水平
//   reward > 0: 多巴胺 > 1（放大 STDP）
//   reward = 0: 多巴胺 = base（中性）
//   reward < 0: 多巴胺 < 1（抑制 STDP）
// -----------------------------------------------------------------------------
void Trainer::apply_reward(float reward) {
    float dopamine;
    if (reward > 0.0f) {
        dopamine = config_.dopamine_base + reward * config_.dopamine_reward;
    } else if (reward < 0.0f) {
        dopamine = config_.dopamine_base + reward * config_.dopamine_penalty;
        dopamine = std::max(dopamine, 0.0f);  // 多巴胺不能为负
    } else {
        dopamine = config_.dopamine_base;
    }
    net_.set_dopamine(dopamine);
}

// -----------------------------------------------------------------------------
// 训练循环
// -----------------------------------------------------------------------------
void Trainer::train() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "训练开始（任务：模式识别）" << std::endl;
    std::cout << "  Episodes:       " << config_.n_episodes << std::endl;
    std::cout << "  Steps/Episode:  " << config_.steps_per_episode << std::endl;
    std::cout << "  Patterns:       " << config_.n_patterns
              << " (target=" << config_.target_pattern << ")" << std::endl;
    std::cout << "  Pattern dur:    " << config_.pattern_duration << " steps" << std::endl;
    std::cout << "========================================\n" << std::endl;

    float* h_input  = new float[N_SENSORY_NEURONS];
    float* h_output = new float[N_MOTOR_NEURONS];

    float running_reward = 0.0f;  // 滑动平均 reward
    const float reward_alpha = 0.05f;

    for (int ep = 0; ep < config_.n_episodes; ep++) {
        net_.reset();
        net_.set_dopamine(config_.dopamine_base);

        float episode_reward_sum = 0.0f;
        int reward_samples = 0;
        int total_spikes = 0;
        int target_hits = 0;        // 目标模式命中数
        int nontarget_correct = 0;  // 非目标模式正确抑制数
        int target_total = 0;
        int nontarget_total = 0;

        for (int step = 0; step < config_.steps_per_episode; step++) {
            // 1. 生成模式输入
            generate_pattern_input(h_input, N_SENSORY_NEURONS, step);

            // 2. 网络步进
            net_.step(h_input, step);

            // 3. 每 25 步（一个模式周期末）采样并给 reward
            if ((step + 1) % config_.pattern_duration == 0) {
                net_.get_output(h_output);

                float reward = compute_reward(h_output, N_MOTOR_NEURONS, step);
                apply_reward(reward);

                episode_reward_sum += reward;
                reward_samples++;
                running_reward = (1.0f - reward_alpha) * running_reward
                               + reward_alpha * reward;

                // 统计命中/误报
                bool is_target = is_target_pattern(step);
                float sum = 0.0f;
                for (int i = 0; i < N_MOTOR_NEURONS; i++) sum += h_output[i];
                float mean_act = sum / N_MOTOR_NEURONS;
                bool is_active = mean_act > config_.motor_active_threshold;

                if (is_target) {
                    target_total++;
                    if (is_active) target_hits++;
                } else {
                    nontarget_total++;
                    if (!is_active) nontarget_correct++;
                }
            }
        }

        NetworkStats stats;
        net_.get_stats(stats);
        episode_count_++;

        float avg_reward = (reward_samples > 0) ? episode_reward_sum / reward_samples : 0.0f;
        float accuracy = (target_total + nontarget_total > 0)
            ? (float)(target_hits + nontarget_correct) / (target_total + nontarget_total)
            : 0.0f;

        if (ep % config_.log_interval == 0 || ep == config_.n_episodes - 1) {
            std::cout << "[Episode " << std::setw(4) << ep << "] "
                      << "Spikes: " << std::setw(6) << stats.total_spikes << " | "
                      << "Exc: " << std::setw(5) << stats.excitatory_spikes << " | "
                      << "Inh: " << std::setw(5) << stats.inhibitory_spikes << " | "
                      << "AvgR: " << std::fixed << std::setprecision(3) << avg_reward << " | "
                      << "Acc: " << std::setprecision(2) << accuracy * 100.0f << "%"
                      << std::endl;
        }

        if (config_.save_interval > 0 && ep > 0 && ep % config_.save_interval == 0) {
            std::string path = config_.checkpoint_dir + "/weights_ep" +
                               std::to_string(ep) + ".bin";
            net_.save_weights(path);
        }
    }

    delete[] h_input;
    delete[] h_output;

    std::cout << "\n训练完成。" << std::endl;
}

// -----------------------------------------------------------------------------
// 评估：单 episode，关闭学习（dopamine=0）
// -----------------------------------------------------------------------------
float Trainer::evaluate() {
    net_.reset();
    net_.set_dopamine(0.0f);  // 评估时关闭 STDP

    float* h_input  = new float[N_SENSORY_NEURONS];
    float* h_output = new float[N_MOTOR_NEURONS];

    int target_hits = 0;
    int nontarget_correct = 0;
    int target_total = 0;
    int nontarget_total = 0;

    for (int step = 0; step < config_.steps_per_episode; step++) {
        generate_pattern_input(h_input, N_SENSORY_NEURONS, step);
        net_.step(h_input, step);

        if ((step + 1) % config_.pattern_duration == 0) {
            net_.get_output(h_output);

            bool is_target = is_target_pattern(step);
            float sum = 0.0f;
            for (int i = 0; i < N_MOTOR_NEURONS; i++) sum += h_output[i];
            float mean_act = sum / N_MOTOR_NEURONS;
            bool is_active = mean_act > config_.motor_active_threshold;

            if (is_target) {
                target_total++;
                if (is_active) target_hits++;
            } else {
                nontarget_total++;
                if (!is_active) nontarget_correct++;
            }
        }
    }

    delete[] h_input;
    delete[] h_output;

    float accuracy = (target_total + nontarget_total > 0)
        ? (float)(target_hits + nontarget_correct) / (target_total + nontarget_total)
        : 0.0f;

    std::cout << "[Eval] Target 命中: " << target_hits << "/" << target_total
              << " | Non-target 正确抑制: " << nontarget_correct << "/" << nontarget_total
              << " | 准确率: " << std::fixed << std::setprecision(2) << accuracy * 100.0f << "%"
              << std::endl;

    return accuracy;
}
