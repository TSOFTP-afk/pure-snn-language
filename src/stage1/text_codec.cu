// =============================================================================
// text_codec.cu - character <-> 5-bit spike pattern codec
// =============================================================================
//
// Encoding scheme (5-bit, 32 chars):
//   Map 32 printable ASCII chars to 5-bit patterns.
//   Set: 'A'-'Z' (26) + ' ' '.' ',' '?' (4)  = 30 chars, plus 2 spare.
//
// char_to_bits(c, bits[5]): returns true if c is encodable
// bits_to_char(bits[5]): returns the decoded char (round each bit to 0/1)
//
// In the SNN autoencoder:
//   - Bits 0..4 are injected as constant current into input neurons 0..4.
//   - Output neurons 5..9 are supervised to reproduce bits 0..4 at final step.
//   - So a successful round-trip means: encode(c) -> SNN -> decode = c.
// =============================================================================

#include "text_codec.cuh"

// 32-char alphabet (5-bit encodable)
// Index 0..31 maps to bit pattern = index itself.
// We pick a friendly printable subset.
static const char kAlphabet[32] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z',' ','.',',','?','!','\''
};

bool char_to_bits(char c, float bits[5]) {
    int idx = -1;
    for (int i = 0; i < 32; i++) {
        if (kAlphabet[i] == c) { idx = i; break; }
    }
    if (idx < 0) return false;
    for (int b = 0; b < 5; b++) {
        bits[b] = (idx >> b) & 1 ? 1.0f : 0.0f;
    }
    return true;
}

char bits_to_char(const float bits[5]) {
    // bits[b] is a firing rate in [0, 1]. Target rates are 0 (bit=0) or 0.5 (bit=1),
    // so threshold at 0.25 (midpoint) for robust classification.
    int idx = 0;
    for (int b = 0; b < 5; b++) {
        int bit = (bits[b] > 0.25f) ? 1 : 0;
        idx |= (bit << b);
    }
    return kAlphabet[idx];
}

int codec_alphabet_size() { return 32; }

char codec_char_at(int idx) {
    if (idx < 0 || idx >= 32) return '?';
    return kAlphabet[idx];
}
