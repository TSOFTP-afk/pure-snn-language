// =============================================================================
// Stage 2e 共激活跟踪采样 kernel 实现 (Task 6)
// =============================================================================
// 设计要点:
//   1. coactivation_sample_kernel: 单 block, 用动态 shared memory 收集当前步
//      发放神经元索引; 每 thread (tid < sample_size) 随机选两个发放神经元
//      形成候选对, 在 tracker 数组中线性查找:
//        - 命中: atomicAdd(coact_count,1) + atomicAdd(modulator_score,current_da)
//                + last_seen = current_step
//        - 未命中且有空位: atomicAdd(d_tracker_count,1) 取槽位, 写入新条目
//      PRNG 为 xorshift32 (与 network_init.cu 一致), per-thread 状态。
//   2. coactivation_prune_kernel: 网格跨步遍历 [0, n_tracked), 清零过期空条目。
// =============================================================================

#include "coactivation_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace stage2e {

namespace {
// 动态 shared memory 发放神经元索引列表上限 (12000 × 4B = 48KB, 单 block 默认上限)
// 超出部分丢弃: 采样用途下不影响统计正确性 (仍是发放神经元的子集)
constexpr int kSharedSpikeCap = 12000;
} // anonymous namespace

// -----------------------------------------------------------------------------
// 共激活采样 kernel
// -----------------------------------------------------------------------------
__global__ void coactivation_sample_kernel(
    CoactTracker* __restrict__ d_trackers,
    int* __restrict__ d_tracker_count,
    const bool* __restrict__ d_spike_flags,
    float current_da,
    int n_neurons,
    int max_trackers,
    int sample_size,
    unsigned int seed,
    int current_step)
{
    extern __shared__ int s_spiked[];          // 动态 shared: 发放神经元索引列表
    __shared__ int s_spiked_count;

    const int tid = threadIdx.x;

    // --- Phase 1: 收集当前步发放神经元索引到 shared memory ---
    if (tid == 0) s_spiked_count = 0;
    __syncthreads();

    for (int i = tid; i < n_neurons; i += blockDim.x) {
        if (d_spike_flags[i]) {
            int slot = atomicAdd(&s_spiked_count, 1);
            if (slot < kSharedSpikeCap) {
                s_spiked[slot] = i;
            }
        }
    }
    __syncthreads();

    int count = s_spiked_count;
    if (count > kSharedSpikeCap) count = kSharedSpikeCap;
    if (count < 2) return;  // 发放神经元不足 2 个, 无法形成候选对

    // --- Phase 2: 每 thread 随机采样一对发放神经元 ---
    if (tid >= sample_size) return;

    // xorshift32 PRNG (与 network_init.cu 一致); 零状态会锁死, 映射到固定非零
    unsigned int rng = seed ^ (tid * 0x9E3779B9u) ^ (current_step * 0x85EBCA6Bu);
    if (rng == 0u) rng = 0x6D2B79F5u;
    auto next_rand = [&]() -> unsigned int {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    int idx1 = static_cast<int>(next_rand() % static_cast<unsigned int>(count));
    int idx2 = static_cast<int>(next_rand() % static_cast<unsigned int>(count));
    // 保证两个不同 (count >= 2 时一定可解)
    while (idx2 == idx1) {
        idx2 = static_cast<int>(next_rand() % static_cast<unsigned int>(count));
    }

    int i = s_spiked[idx1];
    int j = s_spiked[idx2];
    // 规范化 i < j (键唯一)
    if (i > j) { int t = i; i = j; j = t; }

    // 编码 (i, j) -> 32 位键 (n_neurons <= 60000 < 2^16, 16+16 位足够)
    const int key = i | (j << 16);

    // --- Phase 3: 在 tracker 数组中线性查找该候选对 ---
    int n_tracked = *d_tracker_count;
    if (n_tracked > max_trackers) n_tracked = max_trackers;

    int found_idx = -1;
    for (int k = 0; k < n_tracked; ++k) {
        if (d_trackers[k].candidate_pre == key) {
            found_idx = k;
            break;
        }
    }

    if (found_idx >= 0) {
        // 已存在: coact_count++, modulator_score += current_da, last_seen = current_step
        atomicAdd(&d_trackers[found_idx].coact_count, 1);
        atomicAdd(&d_trackers[found_idx].modulator_score, current_da);
        d_trackers[found_idx].last_seen = current_step;
    } else if (n_tracked < max_trackers) {
        // 未命中且有空位: atomicAdd 推进 count, append 新条目
        int slot = atomicAdd(d_tracker_count, 1);
        if (slot < max_trackers) {
            d_trackers[slot].candidate_pre   = key;
            d_trackers[slot].coact_count     = 1;
            d_trackers[slot].last_seen       = current_step;
            d_trackers[slot].modulator_score = current_da;
        } else {
            // 槽位已被并发线程占满, 回滚 count (保持 d_tracker_count <= max_trackers)
            atomicSub(d_tracker_count, 1);
        }
    }
    // 否则 tracker 数组已满, 丢弃该候选对
}

// -----------------------------------------------------------------------------
// 共激活淘汰 kernel
// -----------------------------------------------------------------------------
__global__ void coactivation_prune_kernel(
    CoactTracker* __restrict__ d_trackers,
    int* __restrict__ d_tracker_count,
    int max_trackers,
    int current_step,
    int stale_threshold)
{
    // 仅遍历已使用范围 [0, n_tracked); 超出部分必为空槽
    int n_tracked = *d_tracker_count;
    if (n_tracked > max_trackers) n_tracked = max_trackers;

    // 网格跨步遍历
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_tracked;
         idx += blockDim.x * gridDim.x) {
        CoactTracker t = d_trackers[idx];
        // 空槽位 (candidate_pre==0 且 coact_count==0) 跳过
        if (t.candidate_pre == 0 && t.coact_count == 0) continue;
        // coact_count==0 且超过 stale_threshold 步未更新 -> 淘汰清零
        if (t.coact_count == 0 && (current_step - t.last_seen) > stale_threshold) {
            d_trackers[idx].candidate_pre   = 0;
            d_trackers[idx].coact_count     = 0;
            d_trackers[idx].last_seen       = 0;
            d_trackers[idx].modulator_score = 0.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// P2 修复: 共激活计数衰减 kernel (grid 跨步遍历全部 tracker)
// -----------------------------------------------------------------------------
// 生物学意义: 突触前/后共激活痕迹自然衰退 (Hebbian trace decay)
// 调用时机: 每 COACT_DECAY_INTERVAL 步执行一次
// 效果: coact_count *= COACT_DECAY_FACTOR (0.95)
//       低频共激活对自然归零, 被 coactivation_prune_kernel 淘汰
//       高频共激活对维持 coact_count > form_threshold, 持续参与结构重建
__global__ void coactivation_decay_kernel(
    CoactTracker* __restrict__ d_trackers,
    int* __restrict__ d_tracker_count,
    int max_trackers)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    int n = *d_tracker_count;
    if (n > max_trackers) n = max_trackers;

    for (int idx = tid; idx < n; idx += stride) {
        CoactTracker t = d_trackers[idx];
        if (t.candidate_pre == 0 && t.coact_count == 0) continue;
        // 衰减 coact_count (整数 → 浮点 → 取整)
        float decayed = static_cast<float>(t.coact_count) * COACT_DECAY_FACTOR;
        int new_count = static_cast<int>(decayed + 0.5f);   // 四舍五入
        if (new_count < 0) new_count = 0;
        d_trackers[idx].coact_count = new_count;
        // modulator_score 同步衰减 (保持比例)
        d_trackers[idx].modulator_score *= COACT_DECAY_FACTOR;
    }
}

// -----------------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------------
void launch_coactivation_sample(
    CoactTracker* d_trackers, int* d_tracker_count,
    const bool* d_spike_flags, float current_da,
    int n_neurons, int max_trackers, int sample_size,
    unsigned int seed, int current_step)
{
    // 防御: 缓冲未分配时直接跳过
    if (!d_trackers || !d_tracker_count || !d_spike_flags) return;

    // 单 block: 需 shared memory 共享发放神经元列表, 跨 block 无法共享
    // threads 覆盖 sample_size (向上取整到 256 的倍数, 限制 [256, 1024])
    int threads = ((sample_size + 255) / 256) * 256;
    if (threads < 256)  threads = 256;
    if (threads > 1024) threads = 1024;

    const size_t shared_bytes = kSharedSpikeCap * sizeof(int);
    coactivation_sample_kernel<<<1, threads, shared_bytes>>>(
        d_trackers, d_tracker_count, d_spike_flags, current_da,
        n_neurons, max_trackers, sample_size, seed, current_step);
    CUDA_CHECK_LAST_2E();
}

void launch_coactivation_prune(
    CoactTracker* d_trackers, int* d_tracker_count,
    int max_trackers, int current_step, int stale_threshold)
{
    if (!d_trackers || !d_tracker_count) return;

    int blocks = (max_trackers + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    coactivation_prune_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        d_trackers, d_tracker_count, max_trackers, current_step, stale_threshold);
    CUDA_CHECK_LAST_2E();
}

// P2 修复: 共激活计数衰减 host wrapper
// 调用时机: 每 COACT_DECAY_INTERVAL 步, 在 launch_coactivation_sample 之后执行
void launch_coactivation_decay(
    CoactTracker* d_trackers, int* d_tracker_count, int max_trackers)
{
    if (!d_trackers || !d_tracker_count) return;

    int blocks = (max_trackers + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    if (blocks > 64) blocks = 64;   // 500K 量级, 64 blocks 足够
    coactivation_decay_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        d_trackers, d_tracker_count, max_trackers);
    CUDA_CHECK_LAST_2E();
}

// =============================================================================
// Task 7: 结构可塑性批量重建 kernel
// =============================================================================
// 设计要点:
//   1. structural_rebuild_kernel: 单次 launch 完成阶段1 (候选生成) + 阶段2 (修剪标记)
//      - 阶段1: 网格跨步遍历 tracker, coact_count > θ_form 的候选对解码 (i,j),
//        atomicAdd 抢占 d_new_synapse_count 槽位写入 (上限 max_new)。
//        简化: 跳过 CSR 已有突触检查 (允许冗余); 首过 max_new 而非严格 top-K 排序
//        (严格 top-K 需额外 sort pass, 此处简化以保证 O(n) 单遍扫描)
//      - 阶段2: 网格跨步遍历现有突触, |w| < θ_prune 且 camkii_autophosph < 0.3
//        (CAMKII_AUTOPHOS_FACIL, 未巩固) 的突触在 d_prune_marks 标 1, atomicAdd 计数
//      - 阶段3 (5% 判定): 由 host wrapper 在 sync 后读取两计数器执行
//   2. csr_rebuild_kernel: 单 block 分块原地重建 (正确性优先, 避免双缓冲 640MB)
//      - Phase A: 前向分块迁移存活突触。remap_table[s] <= s (仅前移), 双
//        __syncthreads 迭代保证每轮 "先读后写", 消除并发读写竞争
//      - Phase B: 新突触写入 [surviving_total, new_total), 与 Phase A 目标不重叠
//      - Phase C: 拷贝 d_new_row_ptr → d_row_ptr
// =============================================================================

// 匿名命名空间: 仅本文件使用的辅助 kernel/函数
namespace {

// 统计修剪标记总数 (用于 5% 阈值判定, 避免跳过时 D2H 全量 prune_marks)
__global__ void count_pruned_kernel(const int* __restrict__ marks, int n, int* __restrict__ count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        if (marks[i] != 0) atomicAdd(count, 1);
    }
}

// 初始化新突触 BioSynapse (兴奋性默认参数, 与 network_init.cu 风格一致)
__device__ inline void init_new_synapse(BioSynapse& syn, int pre, int post)
{
    syn.pre_idx            = pre;
    syn.post_idx           = post;
    syn.weight             = 0.1f;            // 小初始权重 (远低于 STDP_W_MAX_2E = 1.5)
    syn.delay_steps        = 1.0f;            // 最小延迟 (柱内 1 步)
    syn.last_pre_spike     = -1.0f;
    syn.last_post_spike    = -1.0f;
    syn.x_pre_trace        = 0.0f;
    syn.x_post_trace       = 0.0f;
    syn.nmda_conductance   = 0.0f;
    syn.ampa_conductance   = 0.0f;
    syn.ca_concentration   = 0.0f;
    syn.resource           = 1.0f;            // STP 资源满
    syn.eligibility        = 0.0f;
    syn.eligibility_slow   = 0.0f;
    syn.utilization        = STP_U_SE;        // 兴奋性基线利用率 (0.2)
    syn.scaling_factor     = 1.0f;
    syn.camkii_autophosph = 0.0f;             // 新突触未巩固
    syn.da_receptor        = DA_RECEPTOR_INIT_EXC;
    syn.ach_receptor       = ACH_RECEPTOR_INIT;
    syn.receptor_flags     = 0x03;            // AMPA | NMDA (兴奋性)
    syn.ne_receptor_u8     = static_cast<uint8_t>(NE_RECEPTOR_INIT * 127.0f);
    syn.ht5_receptor_u8    = static_cast<uint8_t>(HT5_RECEPTOR_INIT * 127.0f);
    syn._pad               = 0;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// 结构重建 kernel (阶段1: 候选生成 + 阶段2: 修剪标记)
// -----------------------------------------------------------------------------
__global__ void structural_rebuild_kernel(
    const CoactTracker* __restrict__ d_trackers,
    int tracker_count,
    int* __restrict__ d_new_synapse_pairs,
    int* __restrict__ d_new_synapse_count,
    float* __restrict__ d_new_modulator_scores,
    const BioSynapse* __restrict__ d_synapses,
    const int* __restrict__ d_row_ptr,
    int n_neurons,
    int* __restrict__ d_prune_marks,
    int* __restrict__ d_prune_count,
    int form_threshold,
    float prune_weight_threshold,
    int max_new)
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // --- 阶段1: 候选生成 (网格跨步遍历 tracker) ---
    // 简化: 跳过 CSR 已有突触检查, 允许冗余; 按 atomicAdd 抢占顺序取前 max_new 个
    // (严格 top-K 按 modulator_score 排序需额外 GPU sort pass, 此处简化为首过 max_new)
    for (int t = tid; t < tracker_count; t += stride) {
        CoactTracker tr = d_trackers[t];
        // 空槽位 (candidate_pre==0 且 coact_count==0) 跳过; 共激活计数需严格大于阈值
        if (tr.candidate_pre == 0 && tr.coact_count == 0) continue;
        if (tr.coact_count > form_threshold) {
            // P1.2 修复: 解码 candidate_pre 并做边界检查
            // candidate_pre = i | (j << 16), 规范化 i < j (与采样端一致)
            int i = tr.candidate_pre & 0xFFFF;        // pre (较小索引)
            int j = tr.candidate_pre >> 16;            // post (较大索引)
            // 边界检查: i, j 必须在 [0, n_neurons) 范围内且 i != j
            // 防止 candidate_pre 含垃圾数据 (如 0xFFFFFFFF) 导致 col_ind 越界
            if (i < 0 || i >= n_neurons || j < 0 || j >= n_neurons || i == j) continue;
            int slot = atomicAdd(d_new_synapse_count, 1);
            if (slot < max_new) {
                d_new_synapse_pairs[2 * slot]     = i;
                d_new_synapse_pairs[2 * slot + 1] = j;
                d_new_modulator_scores[slot]      = tr.modulator_score;
            }
            // 超出 max_new 的候选对丢弃 (atomicAdd 已推进计数, 但不写入)
        }
    }

    // --- 阶段2: 修剪标记 (网格跨步遍历现有突触) ---
    // 总突触数 = d_row_ptr[n_neurons] (CSR 末尾哨兵)
    int n_synapses = d_row_ptr[n_neurons];
    for (int s = tid; s < n_synapses; s += stride) {
        BioSynapse syn = d_synapses[s];
        // |w| < θ_prune 且 CaMKII 未巩固 (< CAMKII_AUTOPHOS_FACIL = 0.3) → 标记修剪
        // 生物学依据: 弱权重 + 未巩固突触最易被修剪 (活动依赖竞争, Huttenlocher 2002)
        if (fabsf(syn.weight) < prune_weight_threshold &&
            syn.camkii_autophosph < CAMKII_AUTOPHOS_FACIL) {
            d_prune_marks[s] = 1;
            atomicAdd(d_prune_count, 1);
        }
    }
    // 阶段3 (5% 判定) 由 host wrapper 在 kernel 完成 + sync 后执行
}

// -----------------------------------------------------------------------------
// CSR 重建 kernel (单 block 分块原地重建)
// -----------------------------------------------------------------------------
// 单 block 设计理由: 多 block 并发会产生读写竞争 (compaction 的目标位置 remap_table[s]
// <= s, 可能与其它线程的源位置重叠)。单 block + 双 __syncthreads 迭代保证每轮
// "先读后写", 消除竞争。10.7M 突触 / 1024 threads ≈ 10.4K 轮, 单 SM 带宽下 ~15ms,
// 在 1000 步重建周期内可接受 (正确性优先)。
// -----------------------------------------------------------------------------
__global__ void csr_rebuild_kernel(
    BioSynapse* __restrict__ d_synapses,
    int* __restrict__ d_row_ptr,
    const int* __restrict__ d_new_synapse_pairs,
    int new_count,
    const int* __restrict__ d_prune_marks,   // 已编码进 remap_table, kernel 内不读取
    int n_neurons,
    int* __restrict__ d_new_row_ptr,
    int* __restrict__ d_remap_table)
{
    (void)d_prune_marks;  // 保留参数以匹配公开签名, 实际不使用

    __shared__ int s_n_old;   // 旧突触总数 (Phase C 会覆盖 d_row_ptr, 必须先保存)
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // --- 读取旧突触总数 ---
    if (tid == 0) {
        s_n_old = d_row_ptr[n_neurons];
    }
    __syncthreads();
    const int n_old = s_n_old;

    // --- Phase A: 前向分块迁移存活突触 ---
    // remap_table[s] >= 0: 存活, 目标位置 = remap_table[s] (<= s, 仅前移)
    // remap_table[s] == -1: 修剪, 跳过
    // 双 sync 迭代: 本轮所有线程先读源 → sync → 写目标 → sync → 下一轮
    // 第 k 轮读写区间 [k*bs, (k+1)*bs): 目标 <= 源, 下一轮源区间不被本轮写覆盖
    for (int s = tid; s < n_old; s += bs) {
        BioSynapse local = d_synapses[s];   // 读源 (先读, 存入 thread-local 变量)
        __syncthreads();                      // 确保本轮所有读完成
        int target = d_remap_table[s];
        if (target >= 0) {
            d_synapses[target] = local;       // 写目标 (后写, target <= s)
        }
        __syncthreads();                      // 确保本轮所有写完成, 下一轮读才安全
    }

    // --- Phase B: 写入新突触 ---
    // 新突触目标位置存于 d_remap_table[n_old + k], 区间 [surviving_total, new_total)
    // 与 Phase A 目标 [0, surviving_total) 不重叠, 安全
    for (int k = tid; k < new_count; k += bs) {
        int target = d_remap_table[n_old + k];
        int pre  = d_new_synapse_pairs[2 * k];
        int post = d_new_synapse_pairs[2 * k + 1];
        BioSynapse fresh;
        init_new_synapse(fresh, pre, post);
        d_synapses[target] = fresh;
    }
    __syncthreads();

    // --- Phase C: 拷贝新 row_ptr → d_row_ptr ---
    for (int i = tid; i <= n_neurons; i += bs) {
        d_row_ptr[i] = d_new_row_ptr[i];
    }
}

// -----------------------------------------------------------------------------
// 结构重建 host wrapper (阶段1+2)
// -----------------------------------------------------------------------------
void launch_structural_rebuild(
    const CoactTracker* d_trackers, int tracker_count,
    int* d_new_pairs, int* d_new_count, float* d_new_scores,
    const BioSynapse* d_synapses, const int* d_row_ptr,
    int n_neurons, int* d_prune_marks, int* d_prune_count,
    int form_thr, float prune_thr, int max_new)
{
    if (!d_trackers || !d_new_pairs || !d_synapses || !d_row_ptr) return;

    // 读取当前突触总数 (用于清零 d_prune_marks 与网格尺寸计算, 单 int D2H)
    int n_synapses = 0;
    CUDA_CHECK_2E(cudaMemcpy(&n_synapses, d_row_ptr + n_neurons,
                             sizeof(int), cudaMemcpyDeviceToHost));

    // 清零计数器与修剪标记 (阶段1/2 使用 atomicAdd, 需从 0 开始)
    CUDA_CHECK_2E(cudaMemset(d_new_count, 0, sizeof(int)));
    if (d_prune_count) CUDA_CHECK_2E(cudaMemset(d_prune_count, 0, sizeof(int)));
    if (d_prune_marks && n_synapses > 0) {
        CUDA_CHECK_2E(cudaMemset(d_prune_marks, 0,
                                 static_cast<size_t>(n_synapses) * sizeof(int)));
    }

    // 网格覆盖 max(tracker_count, n_synapses); 两阶段共享同一网格 (各自网格跨步)
    int n_max = tracker_count > n_synapses ? tracker_count : n_synapses;
    int blocks = (n_max + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    if (blocks > 65535) blocks = 65535;   // CUDA 1D grid 上限

    structural_rebuild_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        d_trackers, tracker_count,
        d_new_pairs, d_new_count, d_new_scores,
        d_synapses, d_row_ptr, n_neurons,
        d_prune_marks, d_prune_count,
        form_thr, prune_thr, max_new);
    CUDA_CHECK_LAST_2E();
    // 注意: 不在此 sync, 调用方 (scheduler) 负责在读取 d_new_synapse_count /
    //       d_prune_count 前同步, 以便决定是否调用 launch_csr_rebuild
}

// -----------------------------------------------------------------------------
// CSR 重建 host wrapper (5% 判定 + CPU 构建 row_ptr/remap_table + GPU 迁移)
// -----------------------------------------------------------------------------
bool launch_csr_rebuild(
    BioSynapse* d_synapses, int* d_row_ptr,
    const int* d_new_pairs, int new_count,
    const int* d_prune_marks, int n_neurons,
    int n_synapses_total,
    int* /*d_new_row_ptr*/, int* /*d_remap_table*/,
    cudaStream_t stream)
{
    if (!d_synapses || !d_row_ptr || !d_prune_marks || n_synapses_total <= 0) {
        return false;
    }

    // --- Step 1: GPU 统计修剪数 (避免跳过时 D2H 全量 prune_marks) ---
    int* d_prune_count = nullptr;
    CUDA_CHECK_2E(cudaMalloc(&d_prune_count, sizeof(int)));
    CUDA_CHECK_2E(cudaMemsetAsync(d_prune_count, 0, sizeof(int), stream));
    {
        int blocks = (n_synapses_total + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
        if (blocks > 65535) blocks = 65535;
        count_pruned_kernel<<<blocks, THREADS_PER_BLOCK_2E, 0, stream>>>(
            d_prune_marks, n_synapses_total, d_prune_count);
    }
    int prune_count = 0;
    CUDA_CHECK_2E(cudaMemcpyAsync(&prune_count, d_prune_count, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));

    // --- Step 2: 5% 阈值判定 ---
    int total_change = new_count + prune_count;
    if (static_cast<float>(total_change) <=
        STRUCTURAL_CHANGE_THRESHOLD * static_cast<float>(n_synapses_total)) {
        CUDA_CHECK_2E(cudaFree(d_prune_count));
        return false;  // 变更不足 5%, 跳过重建
    }

    // --- Step 3: 分配临时缓冲 (remap_table + new_row_ptr) ---
    // remap_table 同时容纳旧突触映射 [0, n_old) 与新突触目标 [n_old, n_old+new_count)
    // 大小: (n_synapses_total + new_count) × 4B + (n_neurons+1) × 4B ≈ 43MB (< 90MB)
    int* d_new_row_ptr = nullptr;
    int* d_remap_table = nullptr;
    const size_t remap_count = static_cast<size_t>(n_synapses_total) + new_count;
    CUDA_CHECK_2E(cudaMalloc(&d_new_row_ptr, (n_neurons + 1) * sizeof(int)));
    CUDA_CHECK_2E(cudaMalloc(&d_remap_table, remap_count * sizeof(int)));

    // --- Step 4: D2H 拷贝 (row_ptr, prune_marks, new_pairs) ---
    std::vector<int> h_row_ptr(n_neurons + 1);
    std::vector<int> h_prune_marks(n_synapses_total);
    std::vector<int> h_new_pairs(new_count > 0 ? 2 * new_count : 0);
    CUDA_CHECK_2E(cudaMemcpyAsync(h_row_ptr.data(), d_row_ptr,
                                  (n_neurons + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_2E(cudaMemcpyAsync(h_prune_marks.data(), d_prune_marks,
                                  static_cast<size_t>(n_synapses_total) * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
    if (new_count > 0) {
        CUDA_CHECK_2E(cudaMemcpyAsync(h_new_pairs.data(), d_new_pairs,
                                      2 * new_count * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));

    const int n_old = h_row_ptr[n_neurons];  // 旧突触总数 (== n_synapses_total)

    // --- Step 5: CPU 构建 new_row_ptr 与 remap_table ---
    // Pass 1: 存活突触映射 (remap_table[s] = 新位置; 修剪的保持 -1)
    std::vector<int> h_surviving_per_row(n_neurons, 0);
    std::vector<int> h_remap_table(n_old + new_count, -1);
    int running = 0;
    for (int r = 0; r < n_neurons; ++r) {
        int surviving_in_row = 0;
        for (int s = h_row_ptr[r]; s < h_row_ptr[r + 1]; ++s) {
            if (h_prune_marks[s] == 0) {
                h_remap_table[s] = running + surviving_in_row;
                ++surviving_in_row;
            }
        }
        h_surviving_per_row[r] = surviving_in_row;
        running += surviving_in_row;
    }
    // 存活突触总数 = running (后续 Pass 3 用于构建 new_row_ptr)

    // Pass 2: 统计每行新突触数 (pre = row)
    std::vector<int> h_new_per_row(n_neurons, 0);
    for (int k = 0; k < new_count; ++k) {
        int pre = h_new_pairs[2 * k];
        if (pre >= 0 && pre < n_neurons) ++h_new_per_row[pre];
    }

    // Pass 3: 构建 new_row_ptr (每行 = 存活 + 新增) 与新突触目标位置
    std::vector<int> h_new_row_ptr(n_neurons + 1);
    h_new_row_ptr[0] = 0;
    for (int r = 0; r < n_neurons; ++r) {
        h_new_row_ptr[r + 1] = h_new_row_ptr[r]
                               + h_surviving_per_row[r]
                               + h_new_per_row[r];
    }
    std::vector<int> h_new_counter(n_neurons, 0);
    for (int k = 0; k < new_count; ++k) {
        int pre = h_new_pairs[2 * k];
        if (pre >= 0 && pre < n_neurons) {
            // 新突触目标 = 该行存活突触之后的位置
            int target = h_new_row_ptr[pre] + h_surviving_per_row[pre] + h_new_counter[pre];
            h_remap_table[n_old + k] = target;
            ++h_new_counter[pre];
        }
    }

    // --- Step 6: H2D 拷贝 (new_row_ptr, remap_table) ---
    CUDA_CHECK_2E(cudaMemcpyAsync(d_new_row_ptr, h_new_row_ptr.data(),
                                  (n_neurons + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_2E(cudaMemcpyAsync(d_remap_table, h_remap_table.data(),
                                  (n_old + new_count) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));

    // --- Step 7: 启动 csr_rebuild_kernel (单 block, 1024 threads) ---
    // 单 block 保证 Phase A 双 sync 迭代的正确性 (消除多 block 并发读写竞争)
    csr_rebuild_kernel<<<1, 1024, 0, stream>>>(
        d_synapses, d_row_ptr, d_new_pairs, new_count,
        d_prune_marks, n_neurons, d_new_row_ptr, d_remap_table);
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));  // 重建完成才能释放临时缓冲

    // --- Step 8: 释放临时缓冲 ---
    CUDA_CHECK_2E(cudaFree(d_new_row_ptr));
    CUDA_CHECK_2E(cudaFree(d_remap_table));
    CUDA_CHECK_2E(cudaFree(d_prune_count));

    return true;
}

// =============================================================================
// Task 19: CSR 完整性运行时校验实现
// =============================================================================
// 设计要点:
//   1. csr_integrity_check_kernel: 网格跨步遍历 [0, n_neurons) 校验 row_ptr 单调性
//      另一网格遍历 [0, n_synapses_expected) 校验 post_idx 范围
//      单线程 atomicOr 校验 row_ptr[n_neurons] == n_synapses_expected
//   2. 错误码用 atomicOr 累积 (幂等, 多线程发现同一错误不会丢失)
//   3. host wrapper 负责清零 d_check_result + 启动 kernel + 同步读取
// =============================================================================

__global__ void csr_integrity_check_kernel(
    const int* __restrict__ d_row_ptr,
    const BioSynapse* __restrict__ d_synapses,
    int n_neurons,
    int n_synapses_expected,
    int* __restrict__ d_check_result)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // --- 检查 1: row_ptr 单调性 (每线程负责若干行) ---
    // row_ptr[i+1] >= row_ptr[i] 对所有 i ∈ [0, n_neurons)
    // 同时校验 row_ptr[0] == 0 (CSR 规范)
    for (int i = tid; i < n_neurons; i += stride) {
        int cur  = d_row_ptr[i];
        int next = d_row_ptr[i + 1];
        if (i == 0 && cur != 0) {
            atomicOr(d_check_result, 1);   // bit0: row_ptr[0] != 0
        }
        if (next < cur) {
            atomicOr(d_check_result, 1);   // bit0: 单调性失败
        }
    }

    // --- 检查 2: col_ind 范围 (用 post_idx 替代, BioSynapse 无独立 col_ind) ---
    // 0 <= synapses[s].post_idx < n_neurons 对所有 s ∈ [0, n_synapses_expected)
    for (int s = tid; s < n_synapses_expected; s += stride) {
        int post = d_synapses[s].post_idx;
        if (post < 0 || post >= n_neurons) {
            atomicOr(d_check_result, 2);   // bit1: col_ind 范围越界
        }
    }

    // --- 检查 3: row_ptr[n_neurons] == n_synapses_expected ---
    // 单线程执行 (避免多线程重复 atomicOr)
    if (tid == 0) {
        int total = d_row_ptr[n_neurons];
        if (total != n_synapses_expected) {
            atomicOr(d_check_result, 4);   // bit2: 总数不一致
        }
    }
}

int launch_csr_integrity_check(
    const int* d_row_ptr,
    const BioSynapse* d_synapses,
    int n_neurons,
    int n_synapses_expected,
    int* d_check_result,
    cudaStream_t stream)
{
    if (!d_row_ptr || !d_synapses || !d_check_result) return -1;
    if (n_neurons <= 0 || n_synapses_expected < 0) return -1;

    // 清零结果缓冲
    CUDA_CHECK_2E(cudaMemsetAsync(d_check_result, 0, sizeof(int), stream));

    // 网格覆盖 max(n_neurons, n_synapses_expected) 的网格跨步遍历
    int n_max = n_neurons > n_synapses_expected ? n_neurons : n_synapses_expected;
    int blocks = (n_max + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    if (blocks > 65535) blocks = 65535;   // CUDA 1D grid 上限

    csr_integrity_check_kernel<<<blocks, THREADS_PER_BLOCK_2E, 0, stream>>>(
        d_row_ptr, d_synapses, n_neurons, n_synapses_expected, d_check_result);
    CUDA_CHECK_LAST_2E();

    // 同步并读取结果
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));
    int result = 0;
    CUDA_CHECK_2E(cudaMemcpyAsync(&result, d_check_result, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));
    return result;
}

// =============================================================================
// Task 19: launch_csr_rebuild_with_integrity_check
//   在 launch_csr_rebuild 完成后立即执行 CSR 完整性校验
//   校验失败时从旧副本回滚 (保留旧 row_ptr + synapses 直到校验通过)
//   返回值: 0=通过 (或跳过重建), !=0=错误码 (回滚后返回)
// =============================================================================
int launch_csr_rebuild_with_integrity_check(
    BioSynapse* d_synapses, int* d_row_ptr,
    const int* d_new_pairs, int new_count,
    const int* d_prune_marks, int n_neurons,
    int n_synapses_total,
    cudaStream_t stream)
{
#if !CSR_INTEGRITY_CHECK_ENABLED
    // 校验禁用: 直接调用 launch_csr_rebuild, 不做校验
    (void)launch_csr_rebuild(d_synapses, d_row_ptr, d_new_pairs, new_count,
                              d_prune_marks, n_neurons, n_synapses_total,
                              nullptr, nullptr, stream);
    return 0;
#else
    // --- Step A: 保存旧 CSR 副本 (用于校验失败时回滚) ---
    // 仅在确实要重建时才保存, 避免无谓显存占用
    // 先调用一次 launch_csr_rebuild 的"预判定"逻辑会重复工作, 故此处直接保存

    // 读取当前 row_ptr[n_neurons] 确认 n_synapses_total 与实际一致
    int cur_total = 0;
    CUDA_CHECK_2E(cudaMemcpyAsync(&cur_total, d_row_ptr + n_neurons,
                                  sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));
    if (cur_total != n_synapses_total) {
        // n_synapses_total 参数与实际不一致, 视为校验失败 (bit2)
        return 4;
    }

    // 分配旧副本缓冲
    BioSynapse* d_old_synapses = nullptr;
    int*        d_old_row_ptr  = nullptr;
    CUDA_CHECK_2E(cudaMalloc(&d_old_synapses,
                              static_cast<size_t>(n_synapses_total) * sizeof(BioSynapse)));
    CUDA_CHECK_2E(cudaMalloc(&d_old_row_ptr,
                              static_cast<size_t>(n_neurons + 1) * sizeof(int)));

    // 拷贝旧数据到副本 (D2D)
    CUDA_CHECK_2E(cudaMemcpyAsync(d_old_synapses, d_synapses,
                                  static_cast<size_t>(n_synapses_total) * sizeof(BioSynapse),
                                  cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK_2E(cudaMemcpyAsync(d_old_row_ptr, d_row_ptr,
                                  static_cast<size_t>(n_neurons + 1) * sizeof(int),
                                  cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));

    // --- Step B: 执行 CSR 重建 ---
    bool rebuilt = launch_csr_rebuild(d_synapses, d_row_ptr, d_new_pairs, new_count,
                                       d_prune_marks, n_neurons, n_synapses_total,
                                       nullptr, nullptr, stream);
    if (!rebuilt) {
        // 变更不足 5%, 跳过重建, 无需校验, 释放旧副本
        CUDA_CHECK_2E(cudaFree(d_old_synapses));
        CUDA_CHECK_2E(cudaFree(d_old_row_ptr));
        return 0;
    }

    // --- Step C: 读取新的 n_synapses (重建后总数可能变化) ---
    int new_n_synapses = 0;
    CUDA_CHECK_2E(cudaMemcpyAsync(&new_n_synapses, d_row_ptr + n_neurons,
                                  sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_2E(cudaStreamSynchronize(stream));

    // --- Step D: 启动 CSR 完整性校验 ---
    int* d_check_result = nullptr;
    CUDA_CHECK_2E(cudaMalloc(&d_check_result, sizeof(int)));

    int err = launch_csr_integrity_check(
        d_row_ptr, d_synapses, n_neurons, new_n_synapses,
        d_check_result, stream);

    if (err == 0) {
        // --- 校验通过: 释放旧副本 ---
        printf("[Stage2e P3-D] CSR 完整性校验通过 (n_synapses=%d→%d)\n",
               n_synapses_total, new_n_synapses);
        CUDA_CHECK_2E(cudaFree(d_old_synapses));
        CUDA_CHECK_2E(cudaFree(d_old_row_ptr));
        CUDA_CHECK_2E(cudaFree(d_check_result));
        return 0;
    } else {
        // --- 校验失败: 从旧副本回滚 ---
        printf("[Stage2e P3-D] ERROR: CSR 完整性校验失败, 已回滚 (错误码=%d, n_old=%d n_new=%d)\n",
               err, n_synapses_total, new_n_synapses);
        // 回滚: 旧副本 → 当前缓冲
        CUDA_CHECK_2E(cudaMemcpyAsync(d_synapses, d_old_synapses,
                                      static_cast<size_t>(n_synapses_total) * sizeof(BioSynapse),
                                      cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_2E(cudaMemcpyAsync(d_row_ptr, d_old_row_ptr,
                                      static_cast<size_t>(n_neurons + 1) * sizeof(int),
                                      cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_2E(cudaStreamSynchronize(stream));
        // 保留旧副本作为正式数据 (实际上已拷贝回 d_synapses/d_row_ptr)
        // 此处释放旧副本缓冲 (数据已回滚到主缓冲)
        CUDA_CHECK_2E(cudaFree(d_old_synapses));
        CUDA_CHECK_2E(cudaFree(d_old_row_ptr));
        CUDA_CHECK_2E(cudaFree(d_check_result));
        return err;
    }
#endif // CSR_INTEGRITY_CHECK_ENABLED
}

} // namespace stage2e
