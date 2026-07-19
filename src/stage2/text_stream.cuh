#ifndef SNN_STAGE2_TEXT_STREAM_CUH
#define SNN_STAGE2_TEXT_STREAM_CUH

#include <string>
#include <cuda_runtime.h>

// =============================================================================
// Stage 2 text stream injector (8-bit / 256 byte, UTF-8 multibyte capable)
// =============================================================================
// Provides a byte stream for unsupervised training. In production this reads
// from DailyDialog / any UTF-8 text file (Chinese, English, mixed). For the
// stage 2a smoke test it falls back to a built-in mini-corpus (now Chinese-
// English mixed to verify multibyte support).
//
// Each call to next_byte() returns one raw byte from the stream; the stream
// wraps around at end-of-text. build_sensory_input(b, buf) fills buf[0..7]
// with the 8-bit encoding of byte b and zeroes buf[8..N_SENSORY_NEURONS-1].
//
// Text filtering policy:
//   - File loading preserves ALL non-zero bytes (UTF-8 multibyte kept as-is)
//   - \r and \n and \t are mapped to space (0x20) for spike-pattern stability
//   - NUL (0x00) is dropped (would terminate C strings and carries no info)
//   - Other control bytes (0x01..0x08, 0x0B..0x0C, 0x0E..0x1F) are kept
//     because UTF-8 never produces them in valid text, so if they appear
//     it is because the file is binary -- caller's responsibility.
// =============================================================================

class TextStream {
public:
    TextStream();
    ~TextStream();

    // Load raw bytes from a file. Returns false on failure.
    // Preserves UTF-8 multibyte sequences; maps \r\n\t to space; drops NUL.
    bool load_from_file(const std::string& path);

    // Use built-in fallback corpus (Chinese-English mixed, for smoke test).
    void load_fallback();

    // Get next byte (wraps around at end). Returns ' ' if stream empty.
    unsigned char next_byte();

    // Build a sensory input vector for a byte.
    // buf must have N_SENSORY_NEURONS floats.
    // Bits 0..7 of b -> neurons 0..7 with amplitude SENSORY_INPUT_GAIN.
    // Neurons 8..N_SENSORY_NEURONS-1 are set to 0.
    void build_sensory_input(unsigned char b, float* buf);

    // Reset read position to 0.
    void reset();

    // Total bytes in the loaded text.
    size_t size() const;

    // Current read position.
    size_t position() const;

private:
    std::string text_;   // raw byte buffer (may contain UTF-8 multibyte)
    size_t      pos_;
};

#endif // SNN_STAGE2_TEXT_STREAM_CUH
