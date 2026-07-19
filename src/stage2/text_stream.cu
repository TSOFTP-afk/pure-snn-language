// =============================================================================
// text_stream.cu - byte stream injector (stage 2, UTF-8 multibyte capable)
// =============================================================================
//
// Host-side byte stream that feeds the SNN sensory cortex. Each step the
// trainer pulls one byte from the stream, encodes it into an 8-bit pattern,
// and uploads the pattern to sensory neurons 0..7 (rest stay at 0).
//
// Supports any UTF-8 text (Chinese, English, mixed). File loading preserves
// multibyte sequences; only \r\n\t are normalized to space and NUL is dropped.
//
// IMPORTANT: fallback corpus uses raw byte arrays (NOT raw Chinese string
// literals) because nvcc's cudafe++ stage parses source files in the system
// locale (GBK on Chinese Windows), which corrupts UTF-8 literals even when
// /utf-8 is passed to the host compiler. This is a hard-won project
// constraint -- see project_memory.md.
// =============================================================================

#include "text_stream.cuh"
#include "text_codec_ext.cuh"
#include "config.h"
#include "../include/config.h"   // N_SENSORY_NEURONS
#include <fstream>
#include <iostream>
#include <cstring>

TextStream::TextStream() : pos_(0) {}
TextStream::~TextStream() {}

bool TextStream::load_from_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[TextStream] Cannot open file: " << path << std::endl;
        return false;
    }

    // Read whole file as raw bytes
    std::string raw((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());

    // Filter:
    //   - \r, \n, \t -> ' ' (0x20)
    //   - NUL (0x00) -> drop
    //   - everything else (including UTF-8 high bytes) -> keep
    text_.clear();
    text_.reserve(raw.size());
    for (unsigned char c : raw) {
        if (c == 0x00) {
            continue;  // drop NUL
        }
        if (c == '\r' || c == '\n' || c == '\t') {
            text_.push_back(' ');
        } else {
            text_.push_back((char)c);
        }
    }

    pos_ = 0;
    std::cout << "[TextStream] Loaded " << path << ": "
              << raw.size() << " raw bytes -> "
              << text_.size() << " filtered bytes" << std::endl;
    return !text_.empty();
}

// -----------------------------------------------------------------------------
// Fallback corpus
// -----------------------------------------------------------------------------
// 12 Chinese dialogues + 13 English dialogues, all encoded as raw byte arrays
// (Chinese) or ASCII string literals (English). Each Chinese char in UTF-8 is
// 3 bytes: 1 head byte (0xE4..0xE9) + 2 continuation bytes (0x80..0xBF).
//
// The Chinese sentences' semantic content (encoded in the byte arrays below):
//   1. Greeting + weather question
//   2. Weather is good, sunny
//   3. Want to walk in the park?
//   4. Sure, I'd love to. Are there flowers in the park?
//   5. Yes, in spring the park is full of cherry blossoms
//   6. Beautiful, I love the scent of cherry blossoms
//   7. We can bring some tea and snacks
//   8. What books do you like to read?
//   9. I like sci-fi and history books
//  10. History books teach us about the past
//  11. Sci-fi books let us imagine the future
//  12. Both are windows to different worlds
//
// Each byte array is NUL-terminated so we can loop until 0x00.
// -----------------------------------------------------------------------------

static inline void append_bytes(std::string& s, const unsigned char* bytes) {
    for (int i = 0; bytes[i] != 0; i++) {
        s.push_back((char)bytes[i]);
    }
}

static inline void append_ascii(std::string& s, const char* ascii) {
    s.append(ascii);
}

void TextStream::load_fallback() {
    text_.clear();

    // ----- Sentence 1: greeting + weather question -----
    {
        const unsigned char s[] = {
            0xE4,0xBD,0xA0, 0xE5,0xA5,0xBD, 0xEF,0xBC,0x8C,
            0xE4,0xBB,0x8A, 0xE5,0xA4,0xA9, 0xE5,0xA4,0xA9,
            0xE6,0xB0,0x94, 0xE6,0x80,0x8E, 0xE4,0xB9,0x88,
            0xE6,0xA0,0xB7, 0xEF,0xBC,0x9F, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 2: weather is good, sunny -----
    {
        const unsigned char s[] = {
            0xE4,0xBB,0x8A, 0xE5,0xA4,0xA9, 0xE5,0xA4,0xA9,
            0xE6,0xB0,0x94, 0xE5,0xBE,0x88, 0xE5,0xA5,0xBD,
            0xEF,0xBC,0x8C, 0xE9,0x98,0xB3, 0xE5,0x85,0x89,
            0xE6,0x98,0x8E, 0xE5,0xAA,0x9A, 0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 3: want to walk in the park? -----
    {
        const unsigned char s[] = {
            0xE4,0xBD,0xA0, 0xE6,0x83,0xB3, 0xE5,0x8E,0xBB,
            0xE5,0x85,0xAC, 0xE5,0x9B,0xAD, 0xE6,0x95,0xA3,
            0xE6,0xAD,0xA5, 0xE5,0x90,0x97, 0xEF,0xBC,0x9F, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 4: sure, I'd love to. flowers in park? -----
    {
        const unsigned char s[] = {
            0xE5,0xA5,0xBD, 0xE7,0x9A,0x84, 0xEF,0xBC,0x8C,
            0xE6,0x88,0x91, 0xE5,0xBE,0x88, 0xE4,0xB9,0x90,
            0xE6,0x84,0x8F, 0xE3,0x80,0x82, 0xE5,0x85,0xAC,
            0xE5,0x9B,0xAD, 0xE9,0x87,0x8C, 0xE6,0x9C,0x89,
            0xE8,0x8A,0xB1, 0xE5,0x90,0x97, 0xEF,0xBC,0x9F, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 5: spring, park full of cherry blossoms -----
    {
        const unsigned char s[] = {
            0xE6,0x9C,0x89, 0xEF,0xBC,0x8C, 0xE6,0x98,0xA5,
            0xE5,0xA4,0xA9, 0xE7,0x9A,0x84, 0xE5,0x85,0xAC,
            0xE5,0x9B,0xAD, 0xE5,0xBC,0x80, 0xE6,0xBB,0xA1,
            0xE4,0xBA,0x86, 0xE6,0xA8,0x9C, 0xE8,0x8A,0xB1,
            0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 6: beautiful, I love cherry blossom scent -----
    {
        const unsigned char s[] = {
            0xE5,0xA4,0xAA, 0xE7,0xBE,0x8E, 0xE4,0xBA,0x86,
            0xEF,0xBC,0x8C, 0xE6,0x88,0x91, 0xE5,0x96,0x9C,
            0xE6,0xAC,0xA2, 0xE6,0xA8,0x9C, 0xE8,0x8A,0xB1,
            0xE7,0x9A,0x84, 0xE9,0xA6,0x99, 0xE5,0x91,0xB3,
            0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 7: we can bring tea and snacks -----
    {
        const unsigned char s[] = {
            0xE6,0x88,0x91, 0xE4,0xBB,0xAC, 0xE5,0x8F,0xAF,
            0xE4,0xBB,0xA5, 0xE5,0xB8,0xA6, 0xE4,0xBA,0x9B,
            0xE8,0x8C,0xB6, 0xE5,0x92,0x8C, 0xE7,0x82,0xB9,
            0xE5,0xBF,0x83, 0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 8: what books do you read? -----
    {
        const unsigned char s[] = {
            0xE4,0xBD,0xA0, 0xE5,0xB9,0xB3, 0xE6,0x97,0xB6,
            0xE5,0x96,0x9C, 0xE6,0xAC,0xA2, 0xE8,0xAF,0xBB,
            0xE4,0xBB,0x80, 0xE4,0xB9,0x88, 0xE4,0xB9,0xA6,
            0xEF,0xBC,0x9F, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 9: I like sci-fi and history books -----
    {
        const unsigned char s[] = {
            0xE6,0x88,0x91, 0xE5,0x96,0x9C, 0xE6,0xAC,0xA2,
            0xE7,0xA7,0x91, 0xE5,0xB9,0xBB, 0xE5,0xB0,0x8F,
            0xE8,0xAF,0xB4, 0xE5,0x92,0x8C, 0xE5,0x8E,0x86,
            0xE5,0x8F,0xB2, 0xE4,0xB9,0xA6, 0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 10: history books teach about the past -----
    {
        const unsigned char s[] = {
            0xE5,0x8E,0x86, 0xE5,0x8F,0xB2, 0xE4,0xB9,0xA6,
            0xE8,0x83,0xBD, 0xE8,0xAE,0xA9, 0xE4,0xBA,0xBA,
            0xE4,0xBA,0x86, 0xE8,0xA7,0xA3, 0xE8,0xBF,0x87,
            0xE5,0x8E,0xBB, 0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 11: sci-fi books let us imagine the future -----
    {
        const unsigned char s[] = {
            0xE7,0xA7,0x91, 0xE5,0xB9,0xBB, 0xE5,0xB0,0x8F,
            0xE8,0xAF,0xB4, 0xE5,0x88,0x99, 0xE8,0xAE,0xA9,
            0xE4,0xBA,0xBA, 0xE6,0x83,0xB3, 0xE8,0xB1,0xA1,
            0xE6,0x9C,0xAA, 0xE6,0x9D,0xA5, 0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }
    // ----- Sentence 12: both are windows to different worlds -----
    {
        const unsigned char s[] = {
            0xE4,0xB8,0xA4, 0xE7,0xA7,0x8D, 0xE4,0xB9,0xA6,
            0xE9,0x83,0xBD, 0xE6,0x98,0xAF, 0xE9,0x80,0x9A,
            0xE5,0xBE,0x80, 0xE4,0xB8,0x8D, 0xE5,0x90,0x8C,
            0xE4,0xB8,0x96, 0xE7,0x95,0x8C, 0xE7,0x9A,0x84,
            0xE7,0xAA,0x97, 0xE6,0x88,0xB7, 0xE3,0x80,0x82, 0x00
        };
        append_bytes(text_, s);
        text_.push_back(' ');
    }

    // ----- English sentences (pure ASCII, no encoding issues) -----
    append_ascii(text_, "Hello, how are you today? ");
    append_ascii(text_, "I am fine, thank you. And you? ");
    append_ascii(text_, "What is your name? ");
    append_ascii(text_, "My name is Alice. Nice to meet you. ");
    append_ascii(text_, "Nice to meet you too. Where are you from? ");
    append_ascii(text_, "I am from Beijing. It is a big city. ");
    append_ascii(text_, "Beijing is beautiful in autumn. ");
    append_ascii(text_, "Do you like reading books? ");
    append_ascii(text_, "I love science fiction and history. ");
    append_ascii(text_, "What do you do for work? ");
    append_ascii(text_, "I am a student. I study physics. ");
    append_ascii(text_, "Goodbye! Have a nice day. ");
    append_ascii(text_, "You too. Bye! ");

    pos_ = 0;
    std::cout << "[TextStream] Loaded fallback corpus: "
              << text_.size() << " bytes (UTF-8 Chinese-English mixed)"
              << std::endl;
}

unsigned char TextStream::next_byte() {
    if (text_.empty()) return ' ';
    unsigned char b = (unsigned char)text_[pos_];
    pos_ = (pos_ + 1) % text_.size();
    return b;
}

void TextStream::build_sensory_input(unsigned char b, float* buf) {
    // 纯 SNN 实验：移除 column_for_byte 硬编码。
    // 输入缓冲区长度 = N_TOTAL_NEURONS（方案 A 保留：sensory 神经元分散在每柱内）
    // 但输入只注入到柱 0 的 sensory 层 [0, 256)。
    //
    // 设计意图：
    //   - 字节 b → 柱 0 的 sensory 神经元 b (one-hot)
    //   - 信号如何传播到柱 1-9 由网络自己决定（通过 STDP + inter-column 突触）
    //   - 如果网络能自发学到字节-柱映射，那是涌现；学不到则反映 SNN 真实能力
    //
    // 与 2A 的区别：
    //   - 2A: column_for_byte(b) 把字节 b 硬编码到柱 (b%10)，是"作弊"
    //   - 纯 SNN: 所有字节都注入柱 0，让网络自己学扩散
    std::memset(buf, 0, N_TOTAL_NEURONS * sizeof(float));

#if ENCODING_MODE_ONEHOT
    // 注入到柱 0 的 sensory 层 (神经元 0..255)
    // 256 个字节 one-hot，落在柱 0 sensory 层 [0, 200) 容量内 (256 > 200, 溢出处理)
    // 实际上 256 > COL_SENSORY_SIZE=200，所以用神经元 0..255（跨 sensory+association 层）
    // 这是可接受的——让 association 层前 56 个神经元也作为直接输入
    buf[(int)b] = SENSORY_INPUT_GAIN;
#else
    // Legacy binary encoding (不再使用)
    float bits[TEXT_CODEC_BITS];
    byte_to_bits_ext(b, bits);
    for (int k = 0; k < TEXT_CODEC_BITS; k++) {
        buf[k] = bits[k] * SENSORY_INPUT_GAIN;
    }
#endif
}

void TextStream::reset() { pos_ = 0; }
size_t TextStream::size() const { return text_.size(); }
size_t TextStream::position() const { return pos_; }
