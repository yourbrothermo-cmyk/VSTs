#pragma once
// Shared constants and helpers. The analysis/vocoder grid matches the
// OpenVPI NSF-HiFiGAN 44.1 kHz / hop 512 / 128-bin release, so the mel and f0
// we compute here can be fed to that vocoder directly.

#include <cmath>
#include <cstddef>
#include <vector>

namespace voxforge {

constexpr int kModelSampleRate = 44100;
constexpr int kHopSize = 512;
constexpr int kFftSize = 2048;
constexpr int kWinSize = 2048;
constexpr int kNumMels = 128;
constexpr float kMelFmin = 40.0f;
constexpr float kMelFmax = 16000.0f;

constexpr double kFrameRate = double(kModelSampleRate) / kHopSize; // ~86.13 fps

inline float hzToMidi(float hz) { return hz > 0.0f ? 69.0f + 12.0f * std::log2(hz / 440.0f) : 0.0f; }
inline float midiToHz(float midi) { return 440.0f * std::exp2((midi - 69.0f) / 12.0f); }
inline double msToFrames(double ms, double frameRate = kFrameRate) { return ms * frameRate / 1000.0; }

// Log-mel spectrogram, row-major [frame][mel bin].
struct MelSpectrogram {
    int numFrames = 0;
    std::vector<float> data;
    float* frame(int t) { return data.data() + size_t(t) * kNumMels; }
    const float* frame(int t) const { return data.data() + size_t(t) * kNumMels; }
};

// Per-frame pitch, one value per hop. f0 == 0 means unvoiced.
struct PitchTrack {
    std::vector<float> f0;
    std::vector<float> confidence;
    size_t size() const { return f0.size(); }
};

} // namespace voxforge
