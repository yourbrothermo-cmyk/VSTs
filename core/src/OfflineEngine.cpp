#include "voxforge/OfflineEngine.h"
#include "voxforge/Resampler.h"
#include <algorithm>
#include <cmath>

namespace voxforge {

Analysis OfflineEngine::analyze(const float* x, size_t n, double sr) const {
    Analysis a;
    a.sourceRate = sr;
    a.sourceLength = n;
    a.audio = resample(x, n, sr, kModelSampleRate);
    PitchTracker tracker;
    a.pitch = tracker.analyze(a.audio.data(), a.audio.size());
    MelExtractor mel;
    a.mel = mel.compute(a.audio.data(), a.audio.size());
    a.notes = segmentNotes(a.pitch.f0);
    a.detectedKey = detectKey(a.notes);
    return a;
}

CorrectionResult OfflineEngine::correct(const Analysis& a, const CorrectionParams& p) {
    return correctPitch(a.pitch.f0, a.notes, p);
}

RenderResult OfflineEngine::render(const Analysis& a, const CorrectionResult& c, const RenderParams& rp,
                                   Vocoder* vocoder, const std::atomic<bool>* cancel, float* progress) {
    RenderResult r;
    const size_t n44 = a.audio.size();
    std::vector<float> wet;
    const bool neural = rp.mode != ResynthMode::Dsp && vocoder != nullptr;

    if (neural) {
        MelSpectrogram mel = a.mel;
        warpFormants(mel, rp.formantSemitones);
        if (!synthesizeChunked(*vocoder, mel, c.f0, wet, r.error, cancel, progress)) return r;
        wet.resize(n44, 0.0f);
        r.vocoderUsed = vocoder->name();

        if (rp.mode == ResynthMode::NeuralHybrid) {
            // Per-frame blend weight: 1 where pitch was moved, 0 where unvoiced or
            // untouched. Smoothed (~25 ms) so switches are inaudible crossfades.
            const int T = int(c.f0.size());
            std::vector<float> w(T, 0.0f);
            for (int t = 0; t < T; ++t)
                if (c.f0[t] > 0.0f)
                    w[t] = std::clamp(std::fabs(c.shiftSemis[t]) * 100.0f / rp.hybridThresholdCents, 0.0f, 1.0f);
            // Dilate by 3 frames so voiced edges are fully covered by the vocoder.
            std::vector<float> d = w;
            for (int t = 0; t < T; ++t)
                for (int k = std::max(0, t - 3); k <= std::min(T - 1, t + 3); ++k) d[t] = std::max(d[t], w[k]);
            const float a1 = std::exp(-1.0f / 2.0f);
            for (int t = 1; t < T; ++t) d[t] = a1 * d[t - 1] + (1 - a1) * d[t];
            for (int t = T - 2; t >= 0; --t) d[t] = std::max(d[t], a1 * d[t + 1] + (1 - a1) * d[t]);
            for (size_t i = 0; i < n44; ++i) {
                const double ft = (double(i) - kHopSize * 0.5) / kHopSize;
                const int t0 = std::clamp(int(std::floor(ft)), 0, std::max(0, T - 1));
                const int t1 = std::min(t0 + 1, std::max(0, T - 1));
                const float fr = std::clamp(float(ft - std::floor(ft)), 0.0f, 1.0f);
                const float g = T > 0 ? d[t0] * (1 - fr) + d[t1] * fr : 0.0f;
                wet[i] = g * wet[i] + (1.0f - g) * a.audio[i];
            }
        }
    } else {
        if (rp.mode != ResynthMode::Dsp && !vocoder) r.error = "no neural model loaded; used DSP fallback";
        wet = psolaShift(a.audio.data(), n44, kModelSampleRate, kHopSize, a.pitch.f0, c.f0);
        r.vocoderUsed = "TD-PSOLA";
        if (progress) *progress = 1.0f;
    }

    r.audio = resample(wet.data(), wet.size(), kModelSampleRate, a.sourceRate);
    r.audio.resize(a.sourceLength, 0.0f);
    r.ok = true;
    return r;
}

} // namespace voxforge
