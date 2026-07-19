// =============================================================================
// text_codec_ext.cu - 8-bit / 256 byte codec (stage 2, UTF-8 multibyte)
// =============================================================================
//
// Encoding: each raw byte b (0x00..0xFF) is directly converted to its 8-bit
// binary representation. Bit k is injected as constant current into sensory
// neuron k (k = 0..7).
//
// This is a strict superset of the previous 7-bit / 95 printable ASCII codec:
//   - ASCII bytes 0x20..0x7E encode exactly as before (high bit always 0)
//   - High bytes (0x80..0xFF) now also encode, enabling UTF-8 multibyte
//     sequences (Chinese, Japanese, emoji, etc.)
//
// Why byte-level instead of codepoint-level:
//   UTF-8 codepoints are 1..4 bytes wide. Fixed-width codepoint encoding
//   needs 21 bits to cover Unicode (0x10FFFF), exceeding our sensory
//   input window. Byte-level encoding keeps the window at 8 neurons and
//   still carries strong structural signal:
//     - ASCII:          0x00..0x7F (high bit = 0)
//     - Chinese head:   0xE0..0xEF (3-byte sequence start)
//     - Continuation:   0x80..0xBF (multi-byte tail)
//   STDP learns temporal correlations at byte granularity, which is well-
//   defined regardless of language.
// =============================================================================

#include "text_codec_ext.cuh"
#include "config.h"

// Encode byte -> 8-bit pattern. Always succeeds.
__host__ void byte_to_bits_ext(unsigned char b, float bits[8]) {
    for (int k = 0; k < TEXT_CODEC_BITS; k++) {
        bits[k] = ((b >> k) & 1) ? 1.0f : 0.0f;
    }
}

// Decode 8 firing rates -> byte. Threshold at 0.25.
__host__ unsigned char bits_to_byte_ext(const float bits[8]) {
    unsigned char b = 0;
    for (int k = 0; k < TEXT_CODEC_BITS; k++) {
        int bit = (bits[k] > 0.25f) ? 1 : 0;
        b |= (unsigned char)(bit << k);
    }
    return b;
}

__host__ int codec_alphabet_size_ext() {
    return TEXT_CODEC_ALPHABET_SIZE;
}
