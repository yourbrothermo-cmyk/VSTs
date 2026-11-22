#pragma once
#include "voxforge/Fft.h"
#include "voxforge/Types.h"
#include <complex>
#include <vector>

namespace voxforge {

// Log-mel front-end matching the NSF-HiFiGAN training recipe:
// 44.1 kHz, n_fft = win = 2048, hop 512, periodic Hann, reflect padding of
// (n_fft - hop) / 2 on both sides, |STFT| = sqrt(re^2 + im^2 + 1e-9),
// 128 Slaney-normalised mel filters 40..16000 Hz, ln(max(x, 1e-5)).
// For an input of n samples this yields floor(n / 512) frames, and the vocoder
// returns frames * 512 samples aligned with the input.
class MelExtractor {
public:
    MelExtractor();

    MelSpectrogram compute(const float* x, size_t n);
    // One frame from kWinSize contiguous samples (already padded/aligned by caller).
    void computeFrame(const float* window2048, float* outMel);

    // Mel-bin centre frequencies (Hz), size kNumMels.
    const std::vector<float>& binCentresHz() const { return centresHz_; }

    static float hzToMelSlaney(float hz);
    static float melToHzSlaney(float mel);

private:
    Fft fft_;
    std::vector<float> window_;
    std::vector<std::vector<std::pair<int, float>>> filters_; // sparse filterbank
    std::vector<float> centresHz_;
    std::vector<std::complex<float>> spec_;
    std::vector<float> mag_;
};

// Shift the spectral envelope encoded in a log-mel spectrogram by `semitones`
// (positive = brighter / smaller vocal tract). Since NSF-HiFiGAN takes f0
// separately, pitch moves while the untouched mel keeps formants in place; this
// lets the user move formants independently on purpose.
void warpFormants(MelSpectrogram& mel, float semitones);

} // namespace voxforge
