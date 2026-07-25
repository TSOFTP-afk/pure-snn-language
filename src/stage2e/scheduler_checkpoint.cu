#include "scheduler.cuh"
#include "input_encoding.cuh"
#include "synapse_kernels.cuh"
#include "neuron_kernels.cuh"
#include "modulatory_kernels.cuh"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace stage2e {
namespace {

constexpr char kMagic[8] = {'S','N','N','2','E','C','P','3'};
constexpr char kFooterMagic[8] = {'S','N','N','2','E','O','K','3'};
constexpr uint32_t kVersion = 3;
constexpr size_t kChunkBytes = 8U * 1024U * 1024U;

struct CheckpointHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t section_count;
    uint32_t reserved;
    uint64_t payload_bytes;
    uint64_t payload_checksum;
    uint32_t n_neurons;
    uint32_t n_synapses;
    uint32_t bio_synapse_bytes;
    uint32_t neuron_state_bytes;
};

struct DiskSection {
    char name[48];
    uint64_t bytes;
};

struct CheckpointFooter {
    char magic[8];
    uint64_t payload_checksum;
};

struct SchedulerState {
    uint32_t state_version;
    uint32_t state_bytes;
    int32_t next_step;
    uint32_t topology_seed;
    uint64_t corpus_size;
    uint64_t corpus_position;
    uint64_t corpus_fingerprint;
    uint8_t e0_ablation;
    uint8_t runtime_state_valid;
    uint8_t reserved[6];
    NetworkStats2e stats;
    int32_t delay_ring_idx;
    int32_t last_phase;
    int32_t total_steps;
    int32_t total_spikes_accum;
    int32_t inject_spikes_accum;
    int32_t min_spikes_per_step;
    int32_t max_spikes_per_step;
    int32_t total_burst_steps;
    int32_t total_single_neuron_burst_spikes;
    int64_t arrived_events_accum;
    int64_t dispatched_events_accum;
    int64_t dropped_events_accum;
    int32_t max_delay_slot_depth;
    int32_t p3_inhibitory_updates;
    int32_t p3_wm_updates;
    float p3_last_activity_drive;
    int32_t p3_kwta_updates;
    int32_t p3_kwta_active_columns;
    int32_t p3_kwta_winner_estimate;
    int32_t p3_kwta_suppressed_estimate;
    int32_t p3_kwta_target_per_column;
    int32_t p3_semantic_eval_updates;
    int32_t p3_semantic_eval_last_step;
    double p3_silhouette_score;
    double p3_js_divergence_mean;
    double p3_js_divergence_max;
    double p3_column_ratio;
    int32_t l6_total_spikes_last;
    float l6_activity_ema_mean;
    double layer_activation_delay[5];
    float layer_chi2_sig_ratio[5];
    double layer_chi2_mean[5];
    float gate_mean;
    float gate_open_ratio;
    DelayQueueCheckpointState delay_queue;
    ModulatoryRuntimeState modulatory_runtime;
};

struct Section {
    const char* name;
    void* pointer;
    uint64_t bytes;
    bool device;
};

uint64_t update_checksum(uint64_t value, const void* data, size_t bytes) {
    static uint32_t table[256]{};
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t x = i;
            for (int bit = 0; bit < 8; ++bit) {
                x = (x >> 1) ^ ((x & 1U) ? 0xEDB88320U : 0U);
            }
            table[i] = x;
        }
        initialized = true;
    }
    const auto* p = static_cast<const unsigned char*>(data);
    uint32_t crc = static_cast<uint32_t>(value) ^ 0xFFFFFFFFU;
    for (size_t i = 0; i < bytes; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFU] ^ (crc >> 8);
    }
    return static_cast<uint64_t>(crc ^ 0xFFFFFFFFU);
}

bool write_exact(FILE* fp, const void* data, size_t bytes) {
    return bytes == 0 || std::fwrite(data, 1, bytes, fp) == bytes;
}

bool read_exact(FILE* fp, void* data, size_t bytes) {
    return bytes == 0 || std::fread(data, 1, bytes, fp) == bytes;
}

bool sync_file(FILE* fp) {
    if (std::fflush(fp) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(fp)) == 0;
#else
    return fsync(fileno(fp)) == 0;
#endif
}

void add(std::vector<Section>* sections, const char* name, void* ptr,
         uint64_t count, uint64_t element_size, bool device = true) {
    sections->push_back({name, ptr, count * element_size, device});
}

} // namespace

struct SchedulerCheckpointAccess {
    static std::vector<Section> make_sections(BioMechanismScheduler* self,
                                              SchedulerState* state);
    static SchedulerState capture_state(BioMechanismScheduler* self, int next_step,
                                        uint32_t seed);
    static bool restore_state(BioMechanismScheduler* self, const SchedulerState& state);
};

std::vector<Section> SchedulerCheckpointAccess::make_sections(
    BioMechanismScheduler* self, SchedulerState* state) {
    PersistentBuffers& b = self->alloc_->buffers();
    std::vector<Section> s;
    s.reserve(55);
    add(&s, "scheduler_state", state, 1, sizeof(*state), false);
    add(&s, "neurons", b.d_neurons, N_TOTAL_NEURONS_2E, sizeof(NeuronStateAdEx));
    add(&s, "spike_flags", b.d_spike_flags, N_TOTAL_NEURONS_2E, sizeof(bool));
    add(&s, "synapses", b.d_synapses, N_TOTAL_SYNAPSES_2E, sizeof(BioSynapse));
    add(&s, "csr_row_ptr", b.d_csr_row_ptr, N_TOTAL_NEURONS_2E + 1ULL, sizeof(int));
    add(&s, "csr_col_idx", b.d_csr_col_idx, N_TOTAL_SYNAPSES_2E, sizeof(int));
    add(&s, "weights_cache", b.d_weights_cache, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "eligibility", b.d_eligibility, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "eligibility_slow", b.d_eligibility_slow, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "synapse_alpha", b.d_synapse_alpha, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "synapse_beta", b.d_synapse_beta, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "pca_W", b.d_pca_W, (uint64_t)N_TOTAL_NEURONS_2E * PATTERN_DIM, sizeof(float));
    add(&s, "ca_snapshot", b.d_ca_snapshot, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "ca_history_sparse", b.d_ca_history_sparse,
        (uint64_t)CA_HISTORY_MAX_ACTIVE * CA_HISTORY_LEN, sizeof(float));
    add(&s, "synapse_delay", b.d_synapse_delay, N_TOTAL_SYNAPSES_2E, sizeof(uint8_t));
    add(&s, "delay_ring_indices", b.d_delay_ring_indices,
        (uint64_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY, sizeof(int));
    add(&s, "delay_ring_current", b.d_delay_ring_current,
        (uint64_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY, sizeof(float));
    add(&s, "stdp_x_pre_trace", b.d_stdp_x_pre_trace, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "camkii_activity", b.d_camkii_activity, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "input_current", b.d_input_current, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "nmda_current", b.d_nmda_current, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "inhibitory_current", b.d_inhibitory_current, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "da_concentration", b.d_da_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "ach_concentration", b.d_ach_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "ne_concentration", b.d_ne_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "ht5_concentration", b.d_ht5_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "hippo_indices", b.d_hippo_indices, HIPP_INDEX_SIZE, sizeof(HippoIndex));
    add(&s, "coact_trackers", b.d_coact_trackers, COACT_TRACKER_SIZE, sizeof(CoactTracker));
    add(&s, "wm_slots", b.d_wm_slots, WM_SLOTS, sizeof(WMSlot));
    add(&s, "subcolumn_fr", b.d_subcolumn_fr, W_VALUE_DIM, sizeof(float));
    add(&s, "baseline_fr", b.d_baseline_fr, W_VALUE_DIM, sizeof(float));
    add(&s, "w_pred", b.d_w_pred, (uint64_t)W_PRED_DIM * W_PRED_DIM, sizeof(float));
    add(&s, "w_value", b.d_w_value, W_VALUE_DIM, sizeof(float));
    add(&s, "pred_fr", b.d_pred_fr, W_PRED_DIM, sizeof(float));
    add(&s, "byte_histogram", b.d_byte_histogram, 256, sizeof(int));
    add(&s, "neuron_byte_counts", b.d_neuron_byte_counts,
        (uint64_t)N_TOTAL_NEURONS_2E * 256, sizeof(int));
    add(&s, "replay_injection", b.d_replay_injection, N_TOTAL_NEURONS_2E, sizeof(float));

    add(&s, "spike_counter", self->d_spike_counter_, 1, sizeof(int));
    add(&s, "single_burst_counter", self->d_single_neuron_burst_counter_, 1, sizeof(int));
    add(&s, "p3_column_spikes", self->d_p3_column_spikes_, N_COLUMNS_2E, sizeof(int));
    add(&s, "p3_kwta_stats", self->d_p3_kwta_stats_, 3, sizeof(int));
    add(&s, "p3_column_byte_responses", self->d_p3_column_byte_responses_,
        (uint64_t)N_COLUMNS_2E * 256, sizeof(int));
    add(&s, "gate_states", self->d_gate_states_, N_COLUMNS_2E, sizeof(ThalamicGateState));
    add(&s, "byte_history", self->d_byte_history_, 256, sizeof(unsigned int));
    add(&s, "gate_stats", self->d_gate_stats_, 4, sizeof(float));
    add(&s, "l6_column_spikes", self->d_l6_column_spikes_, N_COLUMNS_2E, sizeof(int));
    add(&s, "layer_byte_responses", self->d_layer_byte_responses_, 5ULL * 256, sizeof(int));
    add(&s, "layer_sig_count", self->d_layer_sig_count_, 5, sizeof(int));
    add(&s, "layer_act_count", self->d_layer_act_count_, 5, sizeof(int));
    add(&s, "layer_chi2_sum", self->d_layer_chi2_sum_, 5, sizeof(float));
    add(&s, "injections_per_byte", self->d_injections_per_byte_, 256, sizeof(float));
    return s;
}

SchedulerState SchedulerCheckpointAccess::capture_state(
    BioMechanismScheduler* self, int next_step, uint32_t seed) {
    SchedulerState x{};
    x.state_version = 1;
    x.state_bytes = sizeof(x);
    x.next_step = next_step;
    x.topology_seed = seed;
    x.corpus_size = text_corpus_size();
    x.corpus_position = text_stream_position();
    x.corpus_fingerprint = text_corpus_fingerprint();
    x.e0_ablation = self->e0_ablation ? 1 : 0;
    x.stats = self->stats_;
    x.delay_ring_idx = self->delay_ring_idx_;
    x.last_phase = self->last_phase_;
    x.total_steps = self->total_steps_;
    x.total_spikes_accum = self->total_spikes_accum_;
    x.inject_spikes_accum = self->inject_spikes_accum_;
    x.min_spikes_per_step = self->min_spikes_per_step_;
    x.max_spikes_per_step = self->max_spikes_per_step_;
    x.total_burst_steps = self->total_burst_steps_;
    x.total_single_neuron_burst_spikes = self->total_single_neuron_burst_spikes_;
    x.arrived_events_accum = self->arrived_events_accum_;
    x.dispatched_events_accum = self->dispatched_events_accum_;
    x.dropped_events_accum = self->dropped_events_accum_;
    x.max_delay_slot_depth = self->max_delay_slot_depth_;
    x.p3_inhibitory_updates = self->p3_inhibitory_updates_;
    x.p3_wm_updates = self->p3_wm_updates_;
    x.p3_last_activity_drive = self->p3_last_activity_drive_;
    x.p3_kwta_updates = self->p3_kwta_updates_;
    x.p3_kwta_active_columns = self->p3_kwta_active_columns_;
    x.p3_kwta_winner_estimate = self->p3_kwta_winner_estimate_;
    x.p3_kwta_suppressed_estimate = self->p3_kwta_suppressed_estimate_;
    x.p3_kwta_target_per_column = self->p3_kwta_target_per_column_;
    x.p3_semantic_eval_updates = self->p3_semantic_eval_updates_;
    x.p3_semantic_eval_last_step = self->p3_semantic_eval_last_step_;
    x.p3_silhouette_score = self->p3_silhouette_score_;
    x.p3_js_divergence_mean = self->p3_js_divergence_mean_;
    x.p3_js_divergence_max = self->p3_js_divergence_max_;
    x.p3_column_ratio = self->p3_column_ratio_;
    x.l6_total_spikes_last = self->l6_total_spikes_last_;
    x.l6_activity_ema_mean = self->l6_activity_ema_mean_;
    std::copy_n(self->layer_activation_delay_, 5, x.layer_activation_delay);
    std::copy_n(self->layer_chi2_sig_ratio_, 5, x.layer_chi2_sig_ratio);
    std::copy_n(self->layer_chi2_mean_, 5, x.layer_chi2_mean);
    x.gate_mean = self->gate_mean_;
    x.gate_open_ratio = self->gate_open_ratio_;
    x.runtime_state_valid = export_delay_queue_state(&x.delay_queue) ? 1 : 0;
    x.modulatory_runtime = export_modulatory_runtime_state();
    return x;
}

bool SchedulerCheckpointAccess::restore_state(BioMechanismScheduler* self,
                                              const SchedulerState& x) {
    self->e0_ablation = x.e0_ablation != 0;
    self->stats_ = x.stats;
    self->delay_ring_idx_ = x.delay_ring_idx;
    self->last_phase_ = x.last_phase;
    self->total_steps_ = x.total_steps;
    self->total_spikes_accum_ = x.total_spikes_accum;
    self->inject_spikes_accum_ = x.inject_spikes_accum;
    self->min_spikes_per_step_ = x.min_spikes_per_step;
    self->max_spikes_per_step_ = x.max_spikes_per_step;
    self->total_burst_steps_ = x.total_burst_steps;
    self->total_single_neuron_burst_spikes_ = x.total_single_neuron_burst_spikes;
    self->arrived_events_accum_ = x.arrived_events_accum;
    self->dispatched_events_accum_ = x.dispatched_events_accum;
    self->dropped_events_accum_ = x.dropped_events_accum;
    self->max_delay_slot_depth_ = x.max_delay_slot_depth;
    self->p3_inhibitory_updates_ = x.p3_inhibitory_updates;
    self->p3_wm_updates_ = x.p3_wm_updates;
    self->p3_last_activity_drive_ = x.p3_last_activity_drive;
    self->p3_kwta_updates_ = x.p3_kwta_updates;
    self->p3_kwta_active_columns_ = x.p3_kwta_active_columns;
    self->p3_kwta_winner_estimate_ = x.p3_kwta_winner_estimate;
    self->p3_kwta_suppressed_estimate_ = x.p3_kwta_suppressed_estimate;
    self->p3_kwta_target_per_column_ = x.p3_kwta_target_per_column;
    self->p3_semantic_eval_updates_ = x.p3_semantic_eval_updates;
    self->p3_semantic_eval_last_step_ = x.p3_semantic_eval_last_step;
    self->p3_silhouette_score_ = x.p3_silhouette_score;
    self->p3_js_divergence_mean_ = x.p3_js_divergence_mean;
    self->p3_js_divergence_max_ = x.p3_js_divergence_max;
    self->p3_column_ratio_ = x.p3_column_ratio;
    self->l6_total_spikes_last_ = x.l6_total_spikes_last;
    self->l6_activity_ema_mean_ = x.l6_activity_ema_mean;
    std::copy_n(x.layer_activation_delay, 5, self->layer_activation_delay_);
    std::copy_n(x.layer_chi2_sig_ratio, 5, self->layer_chi2_sig_ratio_);
    std::copy_n(x.layer_chi2_mean, 5, self->layer_chi2_mean_);
    self->gate_mean_ = x.gate_mean;
    self->gate_open_ratio_ = x.gate_open_ratio;
    const bool delay_ok = import_delay_queue_state(x.delay_queue);
    import_modulatory_runtime_state(x.modulatory_runtime);
    return delay_ok;
}

namespace {

bool section_layout_matches(const std::vector<Section>& expected,
                            const std::vector<DiskSection>& actual) {
    if (expected.size() != actual.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::strncmp(expected[i].name, actual[i].name, sizeof(actual[i].name)) != 0 ||
            expected[i].bytes != actual[i].bytes) return false;
    }
    return true;
}

} // namespace

int BioMechanismScheduler::save_checkpoint(int next_step, const char* dir,
                                           uint32_t topology_seed) {
    if (!dir || !*dir || next_step < 0) return 1;
    // Epochs are a transient acceleration cache. Persist fully decayed traces
    // so the existing checkpoint schema remains backward compatible.
    materialize_stdp_traces(alloc_, next_step > 0 ? next_step - 1 : 0);
    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::fprintf(stderr, "[Checkpoint] CUDA sync failed: %s\n", cudaGetErrorString(sync_err));
        return 2;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "[Checkpoint] cannot create %s: %s\n", dir, ec.message().c_str());
        return 3;
    }

    const std::filesystem::path final_path =
        std::filesystem::path(dir) / ("ckpt_step" + std::to_string(next_step) + ".snn2e");
    const std::filesystem::path temp_path = final_path.string() + ".tmp";
    if (std::filesystem::exists(final_path, ec)) {
        std::fprintf(stderr, "[Checkpoint] refusing to overwrite %s\n", final_path.string().c_str());
        return 4;
    }
    std::filesystem::remove(temp_path, ec);

    SchedulerState state = SchedulerCheckpointAccess::capture_state(this, next_step, topology_seed);
    if (!state.runtime_state_valid) {
        std::fprintf(stderr, "[Checkpoint] failed to capture delay queue runtime state\n");
        return 5;
    }
    auto sections = SchedulerCheckpointAccess::make_sections(this, &state);
    for (const auto& section : sections) {
        if (!section.pointer && section.bytes > 0) {
            std::fprintf(stderr, "[Checkpoint] null section: %s\n", section.name);
            return 5;
        }
    }
    CheckpointHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kVersion;
    header.header_bytes = sizeof(header);
    header.section_count = static_cast<uint32_t>(sections.size());
    header.n_neurons = N_TOTAL_NEURONS_2E;
    header.n_synapses = N_TOTAL_SYNAPSES_2E;
    header.bio_synapse_bytes = sizeof(BioSynapse);
    header.neuron_state_bytes = sizeof(NeuronStateAdEx);
    for (const auto& section : sections) header.payload_bytes += section.bytes;

    FILE* fp = std::fopen(temp_path.string().c_str(), "wb+");
    if (!fp) return 5;
    bool ok = write_exact(fp, &header, sizeof(header));
    for (const auto& section : sections) {
        DiskSection disk{};
        std::snprintf(disk.name, sizeof(disk.name), "%s", section.name);
        disk.bytes = section.bytes;
        ok = ok && write_exact(fp, &disk, sizeof(disk));
    }

    std::vector<unsigned char> chunk(kChunkBytes);
    uint64_t checksum = 0;
    for (const auto& section : sections) {
        uint64_t offset = 0;
        while (ok && offset < section.bytes) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(chunk.size(), section.bytes - offset));
            if (section.device) {
                const cudaError_t err = cudaMemcpy(chunk.data(),
                    static_cast<const char*>(section.pointer) + offset,
                    count, cudaMemcpyDeviceToHost);
                ok = err == cudaSuccess;
            } else {
                std::memcpy(chunk.data(), static_cast<const char*>(section.pointer) + offset, count);
            }
            if (ok) {
                checksum = update_checksum(checksum, chunk.data(), count);
                ok = write_exact(fp, chunk.data(), count);
            }
            offset += count;
        }
    }

    CheckpointFooter footer{};
    std::memcpy(footer.magic, kFooterMagic, sizeof(kFooterMagic));
    footer.payload_checksum = checksum;
    ok = ok && write_exact(fp, &footer, sizeof(footer));
    header.payload_checksum = checksum;
    ok = ok && std::fseek(fp, 0, SEEK_SET) == 0 && write_exact(fp, &header, sizeof(header));
    ok = ok && sync_file(fp);
    if (std::fclose(fp) != 0) ok = false;
    if (!ok) {
        std::filesystem::remove(temp_path, ec);
        std::fprintf(stderr, "[Checkpoint] write failed for step %d\n", next_step);
        return 6;
    }

    std::filesystem::rename(temp_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(temp_path, ec);
        return 7;
    }
    std::printf("[Checkpoint] saved next_step=%d: %s (%.1f MiB, checksum=%016llx)\n",
                next_step, final_path.string().c_str(),
                header.payload_bytes / (1024.0 * 1024.0),
                static_cast<unsigned long long>(checksum));
    return 0;
}

int BioMechanismScheduler::load_checkpoint(const char* path, int* next_step,
                                           uint32_t* topology_seed) {
    if (!path || !next_step || !topology_seed) return 1;
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return 2;

    CheckpointHeader header{};
    bool ok = read_exact(fp, &header, sizeof(header));
    if (!ok || std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
        header.version != kVersion || header.header_bytes != sizeof(header) ||
        header.n_neurons != N_TOTAL_NEURONS_2E ||
        header.n_synapses != N_TOTAL_SYNAPSES_2E ||
        header.bio_synapse_bytes != sizeof(BioSynapse) ||
        header.neuron_state_bytes != sizeof(NeuronStateAdEx)) {
        std::fclose(fp);
        std::fprintf(stderr, "[Checkpoint] incompatible header: %s\n", path);
        return 3;
    }

    std::vector<DiskSection> disk(header.section_count);
    ok = read_exact(fp, disk.data(), disk.size() * sizeof(DiskSection));
    SchedulerState state{};
    auto sections = SchedulerCheckpointAccess::make_sections(this, &state);
    if (!ok || !section_layout_matches(sections, disk)) {
        std::fclose(fp);
        std::fprintf(stderr, "[Checkpoint] section layout mismatch: %s\n", path);
        return 4;
    }
    uint64_t expected_payload_bytes = 0;
    for (const auto& section : sections) {
        if (!section.pointer && section.bytes > 0) {
            std::fclose(fp);
            return 4;
        }
        expected_payload_bytes += section.bytes;
    }
    if (header.payload_bytes != expected_payload_bytes) {
        std::fclose(fp);
        return 4;
    }

    const long payload_offset = std::ftell(fp);
    std::vector<unsigned char> chunk(kChunkBytes);
    uint64_t checksum = 0;
    uint64_t remaining = header.payload_bytes;
    while (ok && remaining > 0) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(chunk.size(), remaining));
        ok = read_exact(fp, chunk.data(), count);
        if (ok) checksum = update_checksum(checksum, chunk.data(), count);
        remaining -= count;
    }
    CheckpointFooter footer{};
    ok = ok && read_exact(fp, &footer, sizeof(footer));
    if (!ok || checksum != header.payload_checksum ||
        std::memcmp(footer.magic, kFooterMagic, sizeof(kFooterMagic)) != 0 ||
        footer.payload_checksum != checksum) {
        std::fclose(fp);
        std::fprintf(stderr, "[Checkpoint] checksum/completion validation failed: %s\n", path);
        return 5;
    }

    if (std::fseek(fp, payload_offset, SEEK_SET) != 0 ||
        !read_exact(fp, &state, sizeof(state)) ||
        state.state_version != 1 || state.state_bytes != sizeof(state)) {
        std::fclose(fp);
        return 6;
    }
    if (state.corpus_size != text_corpus_size() ||
        state.corpus_fingerprint != text_corpus_fingerprint()) {
        std::fclose(fp);
        std::fprintf(stderr, "[Checkpoint] corpus does not match checkpoint\n");
        return 8;
    }

    if (std::fseek(fp, payload_offset, SEEK_SET) != 0) {
        std::fclose(fp);
        return 6;
    }
    for (const auto& section : sections) {
        uint64_t offset = 0;
        while (ok && offset < section.bytes) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(chunk.size(), section.bytes - offset));
            ok = read_exact(fp, chunk.data(), count);
            if (ok && section.device) {
                const cudaError_t err = cudaMemcpy(
                    static_cast<char*>(section.pointer) + offset,
                    chunk.data(), count, cudaMemcpyHostToDevice);
                ok = err == cudaSuccess;
            } else if (ok) {
                std::memcpy(static_cast<char*>(section.pointer) + offset, chunk.data(), count);
            }
            offset += count;
        }
    }
    std::fclose(fp);
    if (!ok || state.state_version != 1 || state.state_bytes != sizeof(state)) return 7;
    if (!set_text_stream_position(static_cast<size_t>(state.corpus_position))) return 9;
    if (!state.runtime_state_valid || !SchedulerCheckpointAccess::restore_state(this, state)) {
        return 10;
    }
    set_e0_ablation(e0_ablation);
    *next_step = state.next_step;
    *topology_seed = state.topology_seed;
    // Loaded checkpoints contain eager traces materialized after the previous
    // step. Seed the transient lazy epochs without changing the file format.
    reset_stdp_trace_epochs(alloc_, *next_step > 0 ? *next_step - 1 : 0);
    std::printf("[Checkpoint] resumed next_step=%d from %s\n", *next_step, path);
    return 0;
}

int BioMechanismScheduler::prune_checkpoints(const char* dir, int keep_latest) {
    if (!dir || keep_latest <= 0) return 0;
    std::error_code ec;
    struct NumberedCheckpoint {
        long long step;
        std::filesystem::path path;
    };
    std::vector<NumberedCheckpoint> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) return 1;
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.rfind("ckpt_step", 0) == 0 &&
            entry.path().extension() == ".snn2e") {
            const std::string digits = name.substr(9, name.size() - 9 - 6);
            try {
                size_t consumed = 0;
                const long long step = std::stoll(digits, &consumed);
                if (consumed == digits.size()) files.push_back({step, entry.path()});
            } catch (const std::exception&) {
                continue;
            }
        }
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return a.step > b.step;
    });
    for (size_t i = static_cast<size_t>(keep_latest); i < files.size(); ++i) {
        std::filesystem::remove(files[i].path, ec);
        if (ec) return 2;
    }
    return 0;
}

} // namespace stage2e
