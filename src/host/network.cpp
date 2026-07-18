// =============================================================================
// network.cpp - SNN 网络类实现
// =============================================================================

#include "network.h"
#include "neuron.cuh"
#include "synapse.cuh"
#include "stdp.cuh"
#include "network.cuh"
#include "io.cuh"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

SNNNetwork::SNNNetwork(unsigned int seed)
    : seed_(seed), initialized_(false),
      dopamine_level_(1.0f), serotonin_level_(1.0f),
      d_neurons_(nullptr), d_synapses_(nullptr),
      d_row_ptr_(nullptr), d_col_idx_(nullptr),
      d_weights_(nullptr), d_input_current_(nullptr),
      d_spikes_(nullptr), d_external_buf_(nullptr),
      d_stats_(nullptr) {}

SNNNetwork::~SNNNetwork() {
    cleanup();
}

void SNNNetwork::allocate_memory() {
    std::cout << "[Network] 分配 GPU 显存..." << std::endl;

    // 神经元状态
    CUDA_CHECK(cudaMalloc(&d_neurons_, N_TOTAL_NEURONS * sizeof(NeuronState)));

    // 突触
    CUDA_CHECK(cudaMalloc(&d_synapses_, N_TOTAL_SYNAPSES * sizeof(Synapse)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr_, (N_TOTAL_NEURONS + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx_, N_TOTAL_SYNAPSES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights_, N_TOTAL_SYNAPSES * sizeof(float)));

    // 工作缓冲区
    CUDA_CHECK(cudaMalloc(&d_input_current_, N_TOTAL_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spikes_, N_TOTAL_NEURONS * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_external_buf_, N_SENSORY_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stats_, sizeof(NetworkStats)));

    print_memory_usage();
}

void SNNNetwork::free_memory() {
    if (d_neurons_)       cudaFree(d_neurons_);
    if (d_synapses_)      cudaFree(d_synapses_);
    if (d_row_ptr_)       cudaFree(d_row_ptr_);
    if (d_col_idx_)       cudaFree(d_col_idx_);
    if (d_weights_)       cudaFree(d_weights_);
    if (d_input_current_) cudaFree(d_input_current_);
    if (d_spikes_)        cudaFree(d_spikes_);
    if (d_external_buf_)  cudaFree(d_external_buf_);
    if (d_stats_)         cudaFree(d_stats_);

    d_neurons_ = nullptr;
    d_synapses_ = nullptr;
    d_row_ptr_ = nullptr;
    d_col_idx_ = nullptr;
    d_weights_ = nullptr;
    d_input_current_ = nullptr;
    d_spikes_ = nullptr;
    d_external_buf_ = nullptr;
    d_stats_ = nullptr;
}

void SNNNetwork::initialize() {
    if (initialized_) return;

    allocate_memory();

    // 初始化神经元
    init_neurons(d_neurons_, N_TOTAL_NEURONS, seed_);

    // 初始化突触
    init_synapses(d_synapses_, d_row_ptr_, d_col_idx_,
                  N_TOTAL_NEURONS, SYNAPSES_PER_NEURON, seed_ + 1);

    // 同步权重：d_synapses_.weight → d_weights_
    // 突触传播 CSR 用 d_weights_，STDP 用 d_synapses_，必须保持一致
    sync_weights(d_weights_, d_synapses_, N_TOTAL_SYNAPSES);

    // 清零工作缓冲区
    clear_current(d_input_current_, N_TOTAL_NEURONS);
    CUDA_CHECK(cudaMemset(d_spikes_, 0, N_TOTAL_NEURONS * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_stats_, 0, sizeof(NetworkStats)));

    initialized_ = true;
    std::cout << "[Network] 初始化完成" << std::endl;
    std::cout << "  神经元数: " << N_TOTAL_NEURONS << std::endl;
    std::cout << "  突触数:   " << N_TOTAL_SYNAPSES << std::endl;
}

void SNNNetwork::cleanup() {
    if (!initialized_) return;
    free_memory();
    initialized_ = false;
}

void SNNNetwork::step(const float* h_external_input, int time_step) {
    // 1. 清零输入电流累加器
    clear_current(d_input_current_, N_TOTAL_NEURONS);

    // 2. 突触传播：将上一步的脉冲传到突触后电流（CSR 树形规约）
    synaptic_transmission_csr(d_row_ptr_, d_col_idx_, d_weights_,
                              N_TOTAL_NEURONS, d_spikes_, d_input_current_);

    // 3. 注入外部输入（感觉皮层，host → device）
    //    时序：必须在 clear_current 之后、lif_update 之前
    if (h_external_input != nullptr) {
        // 复用 network 持有的 device 缓冲区（避免每步 cudaMalloc）
        CUDA_CHECK(cudaMemcpy(d_external_buf_, h_external_input,
                              N_SENSORY_NEURONS * sizeof(float),
                              cudaMemcpyHostToDevice));
        ::inject_input(d_input_current_, d_external_buf_,
                       N_SENSORY_NEURONS, 1.0f);
    }

    // 4. LIF 神经元更新
    lif_update(d_neurons_, d_input_current_, d_spikes_,
               N_TOTAL_NEURONS, time_step);

    // 5. STDP 权重更新（基于当前脉冲，更新 d_synapses_）
    stdp_update(d_synapses_, N_TOTAL_SYNAPSES,
                d_spikes_, d_spikes_,  // pre 和 post 用同一个数组（全连接视角）
                time_step, dopamine_level_);

    // 6. 同步权重：d_synapses_.weight → d_weights_
    //    CSR 突触传播依赖 d_weights_，STDP 后必须同步
    sync_weights(d_weights_, d_synapses_, N_TOTAL_SYNAPSES);

    // 7. 更新发放率（用于监控）
    update_fire_rate(d_neurons_, d_spikes_, N_TOTAL_NEURONS, 0.95f);

    // 8. 统计
    compute_stats(d_neurons_, d_spikes_, d_stats_, N_TOTAL_NEURONS);
}

void SNNNetwork::reset() {
    reset_neurons(d_neurons_, N_TOTAL_NEURONS);
    clear_current(d_input_current_, N_TOTAL_NEURONS);
    CUDA_CHECK(cudaMemset(d_spikes_, 0, N_TOTAL_NEURONS * sizeof(bool)));
    dopamine_level_ = 1.0f;
    serotonin_level_ = 1.0f;
}

void SNNNetwork::get_output(float* h_output) const {
    int motor_start = N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS;
    // 从 d_neurons_ 的 motor 部分读取【滑动平均发放率】（fire_rate 字段）
    // 而不是当前步的瞬时 spike——后者太稀疏，无法稳定计算 reward
    NeuronState* h_motor = new NeuronState[N_MOTOR_NEURONS];
    CUDA_CHECK(cudaMemcpy(h_motor, d_neurons_ + motor_start,
                          N_MOTOR_NEURONS * sizeof(NeuronState),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < N_MOTOR_NEURONS; i++) {
        h_output[i] = h_motor[i].fire_rate;
    }
    delete[] h_motor;
}

void SNNNetwork::get_stats(NetworkStats& stats) const {
    CUDA_CHECK(cudaMemcpy(&stats, d_stats_, sizeof(NetworkStats),
                          cudaMemcpyDeviceToHost));
}

void SNNNetwork::set_dopamine(float level) {
    dopamine_level_ = level;
}

void SNNNetwork::save_weights(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[Network] 无法打开文件: " << path << std::endl;
        return;
    }

    // 直接从 d_weights_（CSR 权重数组）拷贝到 host
    // d_weights_ 长度 = N_TOTAL_SYNAPSES * sizeof(float) = 4MB
    // 注意：调用方需保证 d_weights_ 与 d_synapses_ 已同步（step 内每步都同步）
    float* h_weights = new float[N_TOTAL_SYNAPSES];
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights_,
                          N_TOTAL_SYNAPSES * sizeof(float),
                          cudaMemcpyDeviceToHost));

    ofs.write(reinterpret_cast<const char*>(h_weights),
              N_TOTAL_SYNAPSES * sizeof(float));

    delete[] h_weights;
    std::cout << "[Network] 权重已保存到 " << path
              << " (" << N_TOTAL_SYNAPSES * sizeof(float) / 1024 / 1024
              << " MB)" << std::endl;
}

void SNNNetwork::load_weights(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[Network] 无法打开文件: " << path << std::endl;
        return;
    }

    float* h_weights = new float[N_TOTAL_SYNAPSES];
    ifs.read(reinterpret_cast<char*>(h_weights),
             N_TOTAL_SYNAPSES * sizeof(float));

    if (ifs.gcount() != N_TOTAL_SYNAPSES * (std::streamsize)sizeof(float)) {
        std::cerr << "[Network] 权重文件大小不匹配，加载失败" << std::endl;
        delete[] h_weights;
        return;
    }

    // 上传到 d_weights_
    CUDA_CHECK(cudaMemcpy(d_weights_, h_weights,
                          N_TOTAL_SYNAPSES * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 反向同步：d_weights_ → d_synapses_.weight
    // 因为 step() 内 STDP 更新 d_synapses_ 后会再同步到 d_weights_，
    // 所以加载后必须让 d_synapses_ 的 weight 字段与 d_weights_ 一致
    // 这里用简单方案：从 d_weights_ 触发同步（需要新增反向同步 kernel）
    // 简化：先直接更新 d_synapses_ 的 weight 字段
    // 用一个临时 host 路径：copy d_synapses_ 到 host，改 weight，再传回
    Synapse* h_synapses = new Synapse[N_TOTAL_SYNAPSES];
    CUDA_CHECK(cudaMemcpy(h_synapses, d_synapses_,
                          N_TOTAL_SYNAPSES * sizeof(Synapse),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < N_TOTAL_SYNAPSES; i++) {
        h_synapses[i].weight = h_weights[i];
    }
    CUDA_CHECK(cudaMemcpy(d_synapses_, h_synapses,
                          N_TOTAL_SYNAPSES * sizeof(Synapse),
                          cudaMemcpyHostToDevice));

    delete[] h_weights;
    delete[] h_synapses;
    std::cout << "[Network] 权重已从 " << path << " 加载" << std::endl;
}

void SNNNetwork::print_memory_usage() const {
    size_t neuron_mem = N_TOTAL_NEURONS * sizeof(NeuronState);
    size_t synapse_mem = N_TOTAL_SYNAPSES * sizeof(Synapse);
    size_t csr_mem = (N_TOTAL_NEURONS + 1) * sizeof(int)
                   + N_TOTAL_SYNAPSES * sizeof(int)
                   + N_TOTAL_SYNAPSES * sizeof(float);
    size_t buf_mem = N_TOTAL_NEURONS * (sizeof(float) + sizeof(bool))
                   + N_SENSORY_NEURONS * sizeof(float)
                   + sizeof(NetworkStats);
    size_t total = neuron_mem + synapse_mem + csr_mem + buf_mem;

    std::cout << "[Memory] 显存占用：" << std::endl;
    std::cout << "  神经元状态:  " << neuron_mem / 1024.0 << " KB" << std::endl;
    std::cout << "  突触:        " << synapse_mem / 1024.0 / 1024.0 << " MB"
              << std::endl;
    std::cout << "  CSR 索引:    " << csr_mem / 1024.0 / 1024.0 << " MB"
              << std::endl;
    std::cout << "  缓冲区:      " << buf_mem / 1024.0 << " KB" << std::endl;
    std::cout << "  总计:        " << total / 1024.0 / 1024.0 << " MB"
              << std::endl;
    std::cout << "  6GB 显存占比: " << (total / (6.0 * 1024 * 1024 * 1024)) * 100
              << "%" << std::endl;
}
