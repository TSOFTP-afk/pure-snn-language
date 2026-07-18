#ifndef SNN_TRAINER_H
#define SNN_TRAINER_H

#include "network.h"
#include <string>

// =============================================================================
// 训练器：管理训练循环和评估
//
// 任务设计（阶段 0.5：模式识别）：
//   - 4 个输入模式 A/B/C/D，每个激活 N_SENSORY_NEURONS/4 的感觉皮层子集
//   - 每个模式持续 25 步，按 A→B→C→D→A... 循环
//   - 目标：Motor 区在模式 A 出现时高活动，其他模式时低活动
//   - Reward：根据当前模式与 Motor 活动的匹配度计算
// =============================================================================

struct TrainingConfig {
    int n_episodes;             // 训练 episode 数
    int steps_per_episode;      // 每个 episode 的步数
    float dopamine_base;        // 基础多巴胺水平（无 reward 时的 STDP 调制）
    float dopamine_reward;      // 正向 reward 时的多巴胺增量
    float dopamine_penalty;     // 负向 reward 时的多巴胺（<1 抑制 STDP）
    float serotonin_base;       // 基础血清素水平
    int log_interval;           // 日志间隔（episode 数）
    int save_interval;          // 保存间隔
    std::string checkpoint_dir; // 检查点目录
    bool enable_monitoring;     // 是否启用监控

    // 任务相关
    int n_patterns;             // 模式数量（默认 4）
    int pattern_duration;       // 每个模式持续的步数（默认 25）
    int target_pattern;         // 目标模式索引（Motor 应在该模式时高活动，默认 0=A）
    float motor_active_threshold; // Motor 平均活动 > 此值视为"高活动"
};

class Trainer {
public:
    Trainer(SNNNetwork& net, TrainingConfig config);
    ~Trainer();

    // 训练入口
    void train();

    // 评估（单 episode）
    float evaluate();

    // 生成指定模式 step 的输入
    void generate_pattern_input(float* h_input, int n_sensory, int step);

    // 计算当前模式（A=0, B=1, C=2, D=3）
    int current_pattern(int step) const {
        return (step / config_.pattern_duration) % config_.n_patterns;
    }

    // 判断当前是否为目标模式
    bool is_target_pattern(int step) const {
        return current_pattern(step) == config_.target_pattern;
    }

private:
    SNNNetwork& net_;
    TrainingConfig config_;

    // 训练日志
    float best_reward_;
    int episode_count_;

    // 计算 reward：当前模式 + Motor 活动 → reward ∈ [-1, 1]
    float compute_reward(const float* motor_output, int n_motor, int step);

    // 根据 reward 设置网络的多巴胺水平
    void apply_reward(float reward);
};

#endif // SNN_TRAINER_H
