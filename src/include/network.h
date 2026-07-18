#ifndef SNN_NETWORK_H
#define SNN_NETWORK_H

#include "types.h"
#include <string>

// =============================================================================
// SNN 网络类：管理 GPU 资源和训练循环
// =============================================================================

class SNNNetwork {
public:
    SNNNetwork(unsigned int seed = 42);
    ~SNNNetwork();

    // 初始化网络（分配显存 + 生成拓扑）
    void initialize();

    // 释放资源
    void cleanup();

    // 单步仿真
    // h_external_input: host 端感觉皮层输入向量（长度 = N_SENSORY_NEURONS）
    //                   传 nullptr 表示本步无外部输入
    void step(const float* h_external_input, int time_step);

    // 重置网络状态（episode 切换）
    void reset();

    // 获取运动皮层输出
    void get_output(float* h_output) const;

    // 获取网络统计
    void get_stats(NetworkStats& stats) const;

    // 设置多巴胺水平（奖励调制）
    void set_dopamine(float level);

    // 保存/加载权重
    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    // 显存使用报告
    void print_memory_usage() const;

private:
    // GPU 内存指针
    NeuronState* d_neurons_;        // [N_TOTAL_NEURONS]
    Synapse*     d_synapses_;       // [N_TOTAL_SYNAPSES]
    int*         d_row_ptr_;        // [N_TOTAL_NEURONS + 1]
    int*         d_col_idx_;        // [N_TOTAL_SYNAPSES]
    float*       d_weights_;        // [N_TOTAL_SYNAPSES]  (CSR 单独权重数组)
    float*       d_input_current_;  // [N_TOTAL_NEURONS]
    bool*        d_spikes_;         // [N_TOTAL_NEURONS]
    float*       d_external_buf_;   // [N_SENSORY_NEURONS] host→device 输入缓冲
    NetworkStats* d_stats_;

    // 调质状态（host 端，每步上传）
    float dopamine_level_;
    float serotonin_level_;

    unsigned int seed_;
    bool initialized_;

    // 辅助方法
    void allocate_memory();
    void free_memory();
};

#endif // SNN_NETWORK_H
