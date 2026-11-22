#pragma once
#include "voxforge/Fft.h"
#include "voxforge/Types.h"
#include <complex>
#include <vector>

namespace voxforge {

struct PitchTrackerConfig {
    double sampleRate = kModelSampleRate;
    int hop = kHopSize;         // frame t is centred at t*hop + hop/2 (matches the mel grid)
    int integrationWindow = 1024;
    float fmin = 65.0f;
    float fmax = 1100.0f;
    float threshold = 0.15f;    // YIN CMND threshold
    float silenceDb = -55.0f;   // frames quieter than this are unvoiced
};

// YIN pitch tracker with an FFT-based difference function, plus offline
// clean-up (octave-error repair, median smoothing, blip removal).
// Not thread-safe: estimateFrame() reuses internal scratch buffers so it is
// allocation-free and safe to call from an audio thread.
class PitchTracker {
public:
    explicit PitchTracker(const PitchTrackerConfig& cfg = {});

    // Whole-buffer analysis on the hop grid. Returns ceil-free floor(n/hop) frames.
    PitchTrack analyze(const float* x, size_t n);

    // Estimate pitch of one analysis frame; `frame` must hold frameSpan() samples.
    float estimateFrame(const float* frame, float& confidence);
    int frameSpan() const { return window_ + tauMax_ + 1; }
    const PitchTrackerConfig& config() const { return cfg_; }

    static void postProcess(PitchTrack& track);

private:
    PitchTrackerConfig cfg_;
    int window_, tauMin_, tauMax_;
    Fft fft_;
    std::vector<std::complex<float>> a_, b_;
    std::vector<float> diff_, cmnd_, energy_;
};

} // namespace voxforge
