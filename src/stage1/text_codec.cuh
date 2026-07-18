#ifndef SNN_STAGE1_TEXT_CODEC_CUH
#define SNN_STAGE1_TEXT_CODEC_CUH

#include <cuda_runtime.h>

// Convert a character to a 5-bit pattern (bits[0..4]).
// Returns true if c is in the 32-char alphabet, false otherwise.
__host__ bool char_to_bits(char c, float bits[5]);

// Convert a 5-bit pattern back to a character.
// Each bit is rounded to 0/1 (threshold 0.5).
__host__ char bits_to_char(const float bits[5]);

// Size of the encodable alphabet (always 32 for 5-bit).
__host__ int codec_alphabet_size();

// Get the character at alphabet index idx (0..31).
__host__ char codec_char_at(int idx);

#endif
