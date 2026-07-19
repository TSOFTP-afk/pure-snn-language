#ifndef SNN_STAGE2_TEXT_CODEC_EXT_CUH
#define SNN_STAGE2_TEXT_CODEC_EXT_CUH

#include <cuda_runtime.h>

// =============================================================================
// Stage 2 text codec: 8-bit / 256 byte (UTF-8 byte stream)
// =============================================================================
// Original stage1 codec was 5-bit / 32 chars; stage2 originally extended to
// 7-bit / 95 printable ASCII. To support Chinese / multilingual corpora we
// now extend further to 8-bit / 256 byte: every byte of the raw UTF-8
// stream is encoded as an 8-bit pattern.
//
// Why byte-level (not codepoint-level):
//   - UTF-8 codepoints vary 1..4 bytes; fixed-width codepoint encoding
//     requires 21 bits to cover all of Unicode (0x10FFFF), exceeding
//     sensory cortex capacity.
//   - Byte-level encoding keeps the input window fixed at 8 neurons and
//     lets the SNN learn the byte-level statistics of UTF-8 (which is
//     itself a meaningful structural signal: e.g. ASCII bytes < 0x80,
//     Chinese head bytes 0xE0..0xEF, continuation bytes 0x80..0xBF).
//   - STDP does not know about characters anyway -- it learns temporal
//     correlations in the spike stream, which are well-defined at byte
//     granularity.
//
// Hard constraint (updated in project_memory.md):
//   - Codec must use 8-bit / 256 byte (UTF-8 byte stream), NOT 7-bit/95
//     printable ASCII (which excluded all non-English text).
//   - Codec functions run on HOST only (__host__).
//   - Encoded bits are injected into sensory neurons 0..7 (8 neurons); the
//     remaining N_SENSORY_NEURONS - 8 = 1992 neurons stay at 0.
//
// Round-trip semantics:
//   byte b -> byte_to_bits_ext(b, bits[8]) -> inject into SNN ->
//   read out 8 firing rates -> bits_to_byte_ext(bits) -> decoded byte.
// A "successful" round-trip means the SNN preserved the bit pattern.
// =============================================================================

// Encode a single byte into 8 bits.
// Always succeeds (all 256 byte values are valid).
// Each bit is written as 1.0f (active) or 0.0f (inactive).
__host__ void byte_to_bits_ext(unsigned char b, float bits[8]);

// Decode 8 firing-rate values back to a byte.
// Each bit is thresholded at 0.25 (midpoint between target 0.0 and 0.5).
__host__ unsigned char bits_to_byte_ext(const float bits[8]);

// Alphabet size (always 256 for 8-bit byte codec).
__host__ int codec_alphabet_size_ext();

#endif // SNN_STAGE2_TEXT_CODEC_EXT_CUH
