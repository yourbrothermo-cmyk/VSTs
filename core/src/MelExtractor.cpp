#include "voxforge/MelExtractor.h"
#include <algorithm>
#include <cmath>

namespace voxforge {

float MelExtractor::hzToMelSlaney(float hz) {
    const float fSp = 200.0f / 3.0f, minLogHz = 1000.0f, minLogMel = minLogHz / fSp;
    const float logStep = std::log(6.4f) / 27.0f;
    return hz < minLogHz ? hz / fSp : minLogMel + std::log(hz / minLogHz) / logStep;
}

float MelExtractor::melToHzSlaney(float mel) {
    const float fSp = 200.0f / 3.0f, minLogHz = 1000.0f, minLogMel = minLogHz / fSp;
    const float logStep = std::log(6.4f) / 27.0f;
    return mel < minLogMel ? mel * fSp : minLogHz * std::exp(logStep * (mel - minLogMel));
}

MelExtractor::MelExtractor() : fft_(kFftSize) {
    window_.resize(kWinSize);
    for (int i = 0; i < kWinSize; ++i)
        window_[i] = float(0.5 - 0.5 * std::cos(2.0 * M_PI * i / kWinSize)); // periodic Hann

    const float mMin = hzToMelSlaney(kMelFmin), mMax = hzToMelSlaney(kMelFmax);
    std::vector<float> hz(kNumMels + 2);
    for (int i = 0; i < kNumMels + 2; ++i)
        hz[i] = melToHzSlaney(mMin + (mMax - mMin) * i / (kNumMels + 1));

    const int bins = kFftSize / 2 + 1;
    filters_.resize(kNumMels);
    centresHz_.resize(kNumMels);
    for (int m = 0; m < kNumMels; ++m) {
        centresHz_[m] = hz[m + 1];
        const float enorm = 2.0f / (hz[m + 2] - hz[m]);
        for (int k = 0; k < bins; ++k) {
            const float f = float(k) * kModelSampleRate / kFftSize;
            const float lower = (f - hz[m]) / (hz[m + 1] - hz[m]);
            const float upper = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
            const float w = std::max(0.0f, std::min(lower, upper));
            if (w > 0.0f) filters_[m].push_back({k, w * enorm});
        }
    }
    spec_.resize(kFftSize);
    mag_.resize(bins);
}

void MelExtractor::computeFrame(const float* x, float* out) {
    for (int i = 0; i < kFftSize; ++i) spec_[i] = {x[i] * window_[i], 0.0f};
    fft_.forward(spec_.data());
    for (size_t k = 0; k < mag_.size(); ++k)
        mag_[k] = std::sqrt(std::norm(spec_[k]) + 1e-9f);
    for (int m = 0; m < kNumMels; ++m) {
        float acc = 0.0f;
        for (const auto& [k, w] : filters_[m]) acc += w * mag_[k];
        out[m] = std::log(std::max(acc, 1e-5f));
    }
}

MelSpectrogram MelExtractor::compute(const float* x, size_t n) {
    MelSpectrogram mel;
    const long pad = (kFftSize - kHopSize) / 2;
    mel.numFrames = int(n / kHopSize);
    mel.data.assign(size_t(mel.numFrames) * kNumMels, std::log(1e-5f));
    std::vector<float> frame(kFftSize);
    auto sample = [&](long i) -> float { // reflect padding (no edge repeat)
        if (n == 0) return 0.0f;
        if (n == 1) return x[0];
        const long N = long(n);
        while (i < 0 || i >= N) {
            if (i < 0) i = -i;
            if (i >= N) i = 2 * (N - 1) - i;
        }
        return x[i];
    };
    for (int t = 0; t < mel.numFrames; ++t) {
        const long start = long(t) * kHopSize - pad;
        for (int i = 0; i < kFftSize; ++i) frame[i] = sample(start + i);
        computeFrame(frame.data(), mel.frame(t));
    }
    return mel;
}

void warpFormants(MelSpectrogram& mel, float semitones) {
    if (std::fabs(semitones) < 1e-3f || mel.numFrames == 0) return;
    const float factor = std::exp2(semitones / 12.0f);
    const float mMin = MelExtractor::hzToMelSlaney(kMelFmin);
    const float mMax = MelExtractor::hzToMelSlaney(kMelFmax);
    const float step = (mMax - mMin) / (kNumMels + 1);

    // For each output bin, read the source envelope at f / factor.
    std::vector<float> srcPos(kNumMels);
    for (int m = 0; m < kNumMels; ++m) {
        const float hz = MelExtractor::melToHzSlaney(mMin + step * (m + 1));
        const float srcMel = MelExtractor::hzToMelSlaney(hz / factor);
        srcPos[m] = (srcMel - mMin) / step - 1.0f;
    }
    std::vector<float> tmp(kNumMels);
    for (int t = 0; t < mel.numFrames; ++t) {
        float* f = mel.frame(t);
        for (int m = 0; m < kNumMels; ++m) {
            const float p = std::clamp(srcPos[m], 0.0f, float(kNumMels - 1));
            const int i0 = std::min(int(p), kNumMels - 2);
            const float fr = p - i0;
            tmp[m] = f[i0] * (1.0f - fr) + f[i0 + 1] * fr;
        }
        std::copy(tmp.begin(), tmp.end(), f);
    }
}

} // namespace voxforge
