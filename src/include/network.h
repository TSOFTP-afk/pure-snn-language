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
    // 注：stage0 用此方法（包含默认 init_synapses 拓扑）。
    // stage2 不调用此方法，改用 allocate_only() + 自定义拓扑生成器。
    void initialize();

    // Stage2 用：仅分配显存 + 初始化神经元 + 清零工作缓冲区。
    // 不调用 init_synapses()，因此不依赖 network_init.cu 的符号。
    // 调用方随后用 columnar_topology.cu 的 init_columnar_synapses() 填充拓扑。
    void allocate_only();

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

    // =====================================================================
    // Stage2 accessors: expose private device pointers so stage2 can build
    // its own columnar topology and run custom training loops.
    // Stage0 never uses these; they exist solely for stage2.
    // =====================================================================
    NeuronState* get_d_neurons()             { return d_neurons_; }
    Synapse*     get_d_synapses()            { return d_synapses_; }
    int*         get_d_row_ptr()             { return d_row_ptr_; }
    int*         get_d_col_idx()             { return d_col_idx_; }
    float*       get_d_weights()             { return d_weights_; }
    float*       get_d_input_current()       { return d_input_current_; }
    bool*        get_d_spikes()              { return d_spikes_; }
    float*       get_d_external_buf()        { return d_external_buf_; }
    NetworkStats* get_d_stats()              { return d_stats_; }
    float        get_dopamine_level() const  { return dopamine_level_; }

    // 标记为已初始化（用于 stage2 在 allocate_only() + 自定义拓扑完成后置位）
    // 注：allocate_only() 已经内部置位 initialized_=true，调用方一般不需要此方法。
    void mark_initialized()                  { initialized_ = true; }

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
