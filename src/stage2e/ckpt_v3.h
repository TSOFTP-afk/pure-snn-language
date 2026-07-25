// =============================================================================
// Stage 2e v3 Checkpoint 读取库 (header-only)
// =============================================================================
// 用于离线工具 (inspect_ckpt / decoder / show_decode_effect) 读取新版 v3 格式
// checkpoint 文件 (与 scheduler_checkpoint.cu 中 save_checkpoint 写入格式对应)
//
// v3 格式布局:
//   1. CheckpointHeader (struct, 固定大小)
//   2. DiskSection[section_count] (section 表, 每个 56 字节)
//   3. payload (各 section 数据连续存储)
//   4. CheckpointFooter (struct, 固定大小)
//
// 关键 section (按名称查找, 不依赖顺序):
//   - "scheduler_state"  : SchedulerState 结构 (含 next_step, topology_seed 等)
//   - "neuron_byte_counts": N_TOTAL_NEURONS_2E × 256 × sizeof(int) 字节响应矩阵
//   - 其他 section 见 scheduler_checkpoint.cu::make_sections
//
// 用法:
//   stage2e::CkptV3Reader reader;
//   if (!reader.open("ckpt_step50000.snn2e")) { ... }
//   auto* nbc = reader.find_section("neuron_byte_counts");
//   if (nbc) {
//       std::vector<int> data(nbc->bytes / sizeof(int));
//       reader.read_section_payload(*nbc, data.data(), data.size() * sizeof(int));
//   }
// =============================================================================

#ifndef SNN_STAGE2E_CKPT_V3_H
#define SNN_STAGE2E_CKPT_V3_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace stage2e {

// 与 scheduler_checkpoint.cu 中的常量保持一致
constexpr char CKPT_V3_MAGIC[8]       = {'S','N','N','2','E','C','P','3'};
constexpr char CKPT_V3_FOOTER_MAGIC[8]= {'S','N','2','E','O','K','3'};
constexpr uint32_t CKPT_V3_VERSION    = 3;

// 与 scheduler_checkpoint.cu::CheckpointHeader 对应 (POD, 跨平台一致)
struct CkptV3Header {
    char     magic[8];
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
static_assert(sizeof(CkptV3Header) == 56, "CkptV3Header 大小必须为 56 字节");

// 与 scheduler_checkpoint.cu::DiskSection 对应
struct CkptV3DiskSection {
    char     name[48];
    uint64_t bytes;
};
static_assert(sizeof(CkptV3DiskSection) == 56, "CkptV3DiskSection 大小必须为 56 字节");

// 与 scheduler_checkpoint.cu::CheckpointFooter 对应
struct CkptV3Footer {
    char     magic[8];
    uint64_t payload_checksum;
};
static_assert(sizeof(CkptV3Footer) == 16, "CkptV3Footer 大小必须为 16 字节");

// 内存中的 section 描述 (含文件偏移)
struct CkptV3SectionInfo {
    std::string name;
    uint64_t    bytes;        // section 数据字节数
    uint64_t    file_offset;  // section 数据在文件中的起始字节偏移
};

// v3 checkpoint 读取器
class CkptV3Reader {
public:
    CkptV3Reader() = default;
    ~CkptV3Reader() { close(); }

    // 禁止拷贝 (持有序言文件句柄)
    CkptV3Reader(const CkptV3Reader&) = delete;
    CkptV3Reader& operator=(const CkptV3Reader&) = delete;

    // 打开 checkpoint 文件并解析头部 + section 表
    // 返回 true 表示成功, false 表示格式错误或 IO 失败 (错误信息已写入 stderr)
    bool open(const std::string& path) {
        close();
        fp_.open(path, std::ios::binary);
        if (!fp_.is_open()) {
            std::fprintf(stderr, "[CkptV3] 无法打开文件: %s\n", path.c_str());
            return false;
        }
        path_ = path;

        // 1. 读头部
        CkptV3Header hdr{};
        if (!read_raw(&hdr, sizeof(hdr))) {
            std::fprintf(stderr, "[CkptV3] 读取头部失败\n");
            return false;
        }
        if (std::memcmp(hdr.magic, CKPT_V3_MAGIC, 8) != 0) {
            std::fprintf(stderr, "[CkptV3] magic 不匹配 (不是 v3 格式?)\n");
            return false;
        }
        if (hdr.version != CKPT_V3_VERSION) {
            std::fprintf(stderr, "[CkptV3] version=%u (期望 %u)\n",
                         hdr.version, CKPT_V3_VERSION);
            return false;
        }
        if (hdr.header_bytes != sizeof(CkptV3Header)) {
            std::fprintf(stderr, "[CkptV3] header_bytes=%u (期望 %zu)\n",
                         hdr.header_bytes, sizeof(CkptV3Header));
            return false;
        }
        header_ = hdr;

        // 2. 读 section 表
        sections_.clear();
        sections_.reserve(hdr.section_count);
        uint64_t payload_offset = sizeof(CkptV3Header) + hdr.section_count * sizeof(CkptV3DiskSection);
        uint64_t running_offset = payload_offset;
        for (uint32_t i = 0; i < hdr.section_count; ++i) {
            CkptV3DiskSection ds{};
            if (!read_raw(&ds, sizeof(ds))) {
                std::fprintf(stderr, "[CkptV3] 读取 section[%u] 失败\n", i);
                return false;
            }
            // name 是固定 48 字节, 以 \0 结尾
            std::string name(ds.name, strnlen(ds.name, sizeof(ds.name)));
            sections_.push_back({name, ds.bytes, running_offset});
            running_offset += ds.bytes;
        }

        // 3. 校验 payload 总大小
        uint64_t section_total = 0;
        for (const auto& s : sections_) section_total += s.bytes;
        if (section_total != header_.payload_bytes) {
            std::fprintf(stderr, "[CkptV3] payload_bytes 不匹配: section 合计=%llu, header=%llu\n",
                         (unsigned long long)section_total,
                         (unsigned long long)header_.payload_bytes);
            return false;
        }

        return true;
    }

    void close() {
        if (fp_.is_open()) fp_.close();
        sections_.clear();
        path_.clear();
        std::memset(&header_, 0, sizeof(header_));
    }

    // 按 section 名查找 (线性扫描, section 数量 ~55, O(n) 足够)
    const CkptV3SectionInfo* find_section(const std::string& name) const {
        for (const auto& s : sections_) {
            if (s.name == name) return &s;
        }
        return nullptr;
    }

    const std::vector<CkptV3SectionInfo>& sections() const { return sections_; }
    const CkptV3Header& header() const { return header_; }
    const std::string& path() const { return path_; }

    // 读取指定 section 的 payload 到调用方缓冲区
    // buf_size 必须不小于 section.bytes
    // 返回 true 表示成功
    bool read_section_payload(const CkptV3SectionInfo& section,
                              void* buf, uint64_t buf_size) {
        if (buf_size < section.bytes) {
            std::fprintf(stderr, "[CkptV3] read_section_payload('%s'): 缓冲区不足 "
                                 "(需要 %llu, 提供 %llu)\n",
                         section.name.c_str(),
                         (unsigned long long)section.bytes,
                         (unsigned long long)buf_size);
            return false;
        }
        fp_.clear();
        fp_.seekg(static_cast<std::streamoff>(section.file_offset), std::ios::beg);
        if (!fp_) {
            std::fprintf(stderr, "[CkptV3] seek 失败: offset=%llu\n",
                         (unsigned long long)section.file_offset);
            return false;
        }
        return read_raw(buf, static_cast<size_t>(section.bytes));
    }

    // 便捷方法: 读取 "scheduler_state" section 并解析出 next_step
    // 返回 true 表示成功 (next_step 写入 *out_next_step)
    bool read_next_step(int* out_next_step) {
        if (out_next_step) *out_next_step = 0;
        // SchedulerState 结构较大且会变化, 这里只读前几个字段
        // 布局 (见 scheduler_checkpoint.cu::SchedulerState):
        //   state_version (u32) + state_bytes (u32) + next_step (i32) + topology_seed (u32)
        const CkptV3SectionInfo* s = find_section("scheduler_state");
        if (!s) {
            std::fprintf(stderr, "[CkptV3] 'scheduler_state' section 未找到\n");
            return false;
        }
        if (s->bytes < 16) {
            std::fprintf(stderr, "[CkptV3] scheduler_state section 太小: %llu\n",
                         (unsigned long long)s->bytes);
            return false;
        }
        uint32_t sv = 0, sb = 0;
        int32_t  ns = 0;
        uint32_t ts = 0;
        if (!read_section_payload(*s, &buffer_, sizeof(buffer_))) return false;
        if (s->bytes < sizeof(buffer_)) {
            // section 比预期小, 仍可读 next_step
        }
        std::memcpy(&sv, buffer_ + 0, 4);
        std::memcpy(&sb, buffer_ + 4, 4);
        std::memcpy(&ns, buffer_ + 8, 4);
        std::memcpy(&ts, buffer_ + 12, 4);
        if (out_next_step) *out_next_step = ns;
        topology_seed_ = ts;
        return true;
    }

    uint32_t topology_seed() const { return topology_seed_; }

private:
    std::ifstream fp_;
    std::string path_;
    CkptV3Header header_{};
    std::vector<CkptV3SectionInfo> sections_;
    uint32_t topology_seed_ = 0;
    // 内部缓冲区用于读 scheduler_state 前 16 字节
    // (布局: state_version + state_bytes + next_step + topology_seed)
    unsigned char buffer_[16];

    bool read_raw(void* buf, size_t bytes) {
        if (bytes == 0) return true;
        fp_.read(static_cast<char*>(buf), static_cast<std::streamsize>(bytes));
        return static_cast<size_t>(fp_.gcount()) == bytes;
    }
};

} // namespace stage2e

#endif // SNN_STAGE2E_CKPT_V3_H
