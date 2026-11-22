#include "voxforge/PitchTracker.h"
#include <algorithm>
#include <cmath>

namespace voxforge {

namespace {
int nextPow2(int v) { int p = 1; while (p < v) p <<= 1; return p; }
}

PitchTracker::PitchTracker(const PitchTrackerConfig& cfg)
    : cfg_(cfg),
      window_(int(std::lround(cfg.integrationWindow * cfg.sampleRate / kModelSampleRate))),
      tauMin_(std::max(2, int(std::floor(cfg.sampleRate / cfg.fmax)))),
      tauMax_(int(std::ceil(cfg.sampleRate / cfg.fmin))),
      fft_(nextPow2(window_ + tauMax_ + 1 + window_)) {
    a_.resize(fft_.size());
    b_.resize(fft_.size());
    diff_.resize(tauMax_ + 2);
    cmnd_.resize(tauMax_ + 2);
    energy_.resize(frameSpan() + 1);
}

float PitchTracker::estimateFrame(const float* x, float& confidence) {
    confidence = 0.0f;
    const int span = frameSpan();
    const int W = window_;

    // Silence gate on the integration window.
    double e0 = 0.0;
    for (int j = 0; j < W; ++j) e0 += double(x[j]) * x[j];
    const double rms = std::sqrt(e0 / W);
    if (20.0 * std::log10(rms + 1e-12) < cfg_.silenceDb) return 0.0f;

    // r(tau) = sum_{j<W} x[j] x[j+tau] via FFT cross-correlation.
    const int N = fft_.size();
    for (int i = 0; i < N; ++i) {
        a_[i] = i < W ? x[i] : 0.0f;
        b_[i] = i < span ? x[i] : 0.0f;
    }
    fft_.forward(a_.data());
    fft_.forward(b_.data());
    for (int i = 0; i < N; ++i) b_[i] = std::conj(a_[i]) * b_[i];
    fft_.inverse(b_.data());
    const float invN = 1.0f / N;

    // Sliding energy of x[tau .. tau+W).
    energy_[0] = 0.0f;
    for (int i = 0; i < span; ++i) energy_[i + 1] = energy_[i] + x[i] * x[i];

    diff_[0] = 0.0f;
    float running = 0.0f;
    cmnd_[0] = 1.0f;
    for (int tau = 1; tau <= tauMax_; ++tau) {
        const float et = energy_[tau + W] - energy_[tau];
        const float d = std::max(0.0f, float(e0) + et - 2.0f * b_[tau].real() * invN);
        diff_[tau] = d;
        running += d;
        cmnd_[tau] = running > 0.0f ? d * tau / running : 1.0f;
    }

    int best = -1;
    for (int tau = tauMin_; tau <= tauMax_; ++tau) {
        if (cmnd_[tau] < cfg_.threshold) {
            while (tau + 1 <= tauMax_ && cmnd_[tau + 1] < cmnd_[tau]) ++tau;
            best = tau;
            break;
        }
    }
    if (best < 0) {
        // No dip under threshold: take the global minimum but report low confidence.
        best = tauMin_;
        for (int tau = tauMin_; tau <= tauMax_; ++tau)
            if (cmnd_[tau] < cmnd_[best]) best = tau;
        confidence = std::max(0.0f, 1.0f - cmnd_[best]);
        if (cmnd_[best] > 2.0f * cfg_.threshold) return 0.0f;
    } else {
        confidence = std::max(0.0f, 1.0f - cmnd_[best]);
    }

    // Parabolic interpolation around the minimum.
    float tauF = float(best);
    if (best > 1 && best < tauMax_) {
        const float s0 = cmnd_[best - 1], s1 = cmnd_[best], s2 = cmnd_[best + 1];
        const float denom = s0 - 2.0f * s1 + s2;
        if (std::fabs(denom) > 1e-9f) tauF += 0.5f * (s0 - s2) / denom;
    }
    return float(cfg_.sampleRate / tauF);
}

PitchTrack PitchTracker::analyze(const float* x, size_t n) {
    PitchTrack track;
    const int hop = cfg_.hop;
    const size_t frames = n / size_t(hop);
    track.f0.assign(frames, 0.0f);
    track.confidence.assign(frames, 0.0f);
    const int span = frameSpan();
    std::vector<float> buf(span);
    for (size_t t = 0; t < frames; ++t) {
        const long centre = long(t) * hop + hop / 2;
        const long start = centre - span / 2;
        for (int i = 0; i < span; ++i) {
            const long idx = start + i;
            buf[i] = (idx >= 0 && size_t(idx) < n) ? x[idx] : 0.0f;
        }
        float conf = 0.0f;
        track.f0[t] = estimateFrame(buf.data(), conf);
        track.confidence[t] = conf;
    }
    postProcess(track);
    return track;
}

void PitchTracker::postProcess(PitchTrack& track) {
    auto& f0 = track.f0;
    const int n = int(f0.size());
    if (n == 0) return;

    // 1) Octave-error repair against a local median of voiced neighbours.
    std::vector<float> fixed = f0, neigh;
    for (int t = 0; t < n; ++t) {
        if (f0[t] <= 0.0f) continue;
        neigh.clear();
        for (int k = std::max(0, t - 6); k <= std::min(n - 1, t + 6); ++k)
            if (k != t && f0[k] > 0.0f) neigh.push_back(f0[k]);
        if (neigh.size() < 4) continue;
        std::nth_element(neigh.begin(), neigh.begin() + neigh.size() / 2, neigh.end());
        const float med = neigh[neigh.size() / 2];
        const float ratio = f0[t] / med;
        if (std::fabs(ratio - 2.0f) < 0.2f) fixed[t] = f0[t] * 0.5f;
        else if (std::fabs(ratio - 0.5f) < 0.05f) fixed[t] = f0[t] * 2.0f;
    }
    f0 = fixed;

    // 2) Fill single-frame unvoiced holes inside voiced regions.
    for (int t = 1; t + 1 < n; ++t)
        if (f0[t] <= 0.0f && f0[t - 1] > 0.0f && f0[t + 1] > 0.0f)
            f0[t] = std::sqrt(f0[t - 1] * f0[t + 1]);

    // 3) Remove voiced blips shorter than 3 frames.
    for (int t = 0; t < n;) {
        if (f0[t] <= 0.0f) { ++t; continue; }
        int e = t;
        while (e < n && f0[e] > 0.0f) ++e;
        if (e - t < 3) for (int k = t; k < e; ++k) f0[k] = 0.0f;
        t = e;
    }

    // 4) 3-tap median inside voiced runs (keeps vibrato, removes spikes).
    fixed = f0;
    for (int t = 1; t + 1 < n; ++t) {
        if (f0[t - 1] > 0 && f0[t] > 0 && f0[t + 1] > 0) {
            float a = f0[t - 1], b = f0[t], c = f0[t + 1];
            fixed[t] = std::max(std::min(a, b), std::min(std::max(a, b), c));
        }
    }
    f0 = fixed;
}

} // namespace voxforge
