#ifndef SNN_IO_CUH
#define SNN_IO_CUH

#include "types.h"

// =============================================================================
// IO kernel 的 host 端包装函数声明
// =============================================================================

// 将外部输入注入到感觉皮层
void inject_input(float* d_input, const float* d_external,
                  int n_sensory, float gain);

// 从运动皮层读取输出
void read_output(const bool* d_spikes, float* d_output,
                 int n_motor, int motor_start);

// 统计网络状态（用于监控）
// d_weights / n_synapses 传 nullptr/0 时跳过 mean_weight 计算（向后兼容）
void compute_stats(const NeuronState* d_neurons, const bool* d_spikes,
                   NetworkStats* d_stats, int n_neurons,
                   const float* d_weights = nullptr, int n_synapses = 0);

#endif // SNN_IO_CUH
