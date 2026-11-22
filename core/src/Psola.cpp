#include "voxforge/Vocoder.h"
#include <algorithm>
#include <cmath>

namespace voxforge {

namespace {
// f0 at sample position s; frame t is centred at t*hop + hop/2.
float f0At(const std::vector<float>& f0, double s, int hop) {
    if (f0.empty()) return 0.0f;
    const double ft = (s - hop * 0.5) / hop;
    const long t0 = long(std::floor(ft));
    const long n = long(f0.size());
    auto get = [&](long t) { return f0[size_t(std::clamp(t, 0L, n - 1))]; };
    const float a = get(t0), b = get(t0 + 1);
    if (a <= 0.0f || b <= 0.0f) return (ft - t0 < 0.5) ? a : b;
    const float fr = float(ft - t0);
    return a * (1.0f - fr) + b * fr;
}
} // namespace

std::vector<float> psolaShift(const float* x, size_t n, double sr, int hop, const std::vector<float>& f0Src,
                              const std::vector<float>& f0Dst) {
    std::vector<float> out(n, 0.0f);
    if (n == 0) return out;
    const double unvoicedPeriod = 0.0058 * sr;

    // 1) Analysis pitch marks, snapped to local waveform peaks for consistency.
    std::vector<double> marks;
    std::vector<double> periods;
    double s = 0.0;
    while (s < double(n)) {
        const float f = f0At(f0Src, s, hop);
        double P = f > 0.0f ? sr / f : unvoicedPeriod;
        double m = s;
        if (f > 0.0f && !marks.empty()) {
            const long lo = std::max(0L, long(s - 0.25 * P)), hi = std::min(long(n) - 1, long(s + 0.25 * P));
            long best = long(s);
            for (long k = lo; k <= hi; ++k)
                if (x[k] > x[std::clamp(best, 0L, long(n) - 1)]) best = k;
            m = double(best);
            if (m <= marks.back()) m = marks.back() + 0.5 * P;
        }
        marks.push_back(m);
        periods.push_back(P);
        s = m + P;
    }

    // 2) Synthesis: place grains at target-period spacing, each taken from the
    // analysis mark nearest in time (1:1 time map).
    double ts = marks.front();
    size_t idx = 0;
    while (ts < double(n)) {
        const float fs = f0At(f0Src, ts, hop), fd = f0At(f0Dst, ts, hop);
        while (idx + 1 < marks.size() && std::fabs(marks[idx + 1] - ts) <= std::fabs(marks[idx] - ts)) ++idx;
        const double Pa = periods[idx];
        double Ps = (fs > 0.0f && fd > 0.0f) ? sr / fd : Pa;
        Ps = std::max(Ps, 16.0);
        const double gain = std::min(2.0, Ps / Pa); // keep level constant as grain density changes
        const long ma = long(std::lround(marks[idx]));
        const long L = long(std::ceil(Pa));
        const long centre = long(std::lround(ts));
        for (long k = -L; k <= L; ++k) {
            const long src = ma + k, dst = centre + k;
            if (src < 0 || src >= long(n) || dst < 0 || dst >= long(n)) continue;
            const double w = 0.5 + 0.5 * std::cos(M_PI * double(k) / double(L + 1));
            out[size_t(dst)] += float(gain * w * x[src]);
        }
        ts += Ps;
    }
    return out;
}

bool synthesizeChunked(Vocoder& v, const MelSpectrogram& mel, const std::vector<float>& f0, std::vector<float>& out,
                       std::string& error, const std::atomic<bool>* cancel, float* progress, int chunkFrames, int overlap) {
    const int T = mel.numFrames;
    out.assign(size_t(T) * kHopSize, 0.0f);
    if (T == 0) return true;
    std::vector<float> chunkOut;
    int prevEnd = 0; // frames already written definitively
    for (int start = 0; start < T;) {
        if (cancel && cancel->load()) { error = "cancelled"; return false; }
        const int ctxStart = std::max(0, start - overlap);
        const int end = std::min(T, start + chunkFrames);
        const int ctxEnd = std::min(T, end + overlap);
        const int frames = ctxEnd - ctxStart;
        if (!v.synthesize(mel.frame(ctxStart), frames, f0.data() + ctxStart, chunkOut, error)) return false;
        chunkOut.resize(size_t(frames) * kHopSize, 0.0f);

        // Crossfade the first `overlap` frames of this chunk with what is already there.
        const size_t fadeLen = size_t(start - ctxStart) * kHopSize;
        const size_t base = size_t(ctxStart) * kHopSize;
        for (size_t i = 0; i < size_t(end - ctxStart) * kHopSize; ++i) {
            const size_t o = base + i;
            if (i < fadeLen && o < size_t(prevEnd) * kHopSize) {
                const float w = float(i + 1) / float(fadeLen + 1);
                out[o] = out[o] * (1.0f - w) + chunkOut[i] * w;
            } else {
                out[o] = chunkOut[i];
            }
        }
        // Keep the right context so the next chunk can crossfade into it.
        for (size_t i = size_t(end - ctxStart) * kHopSize; i < chunkOut.size(); ++i)
            out[base + i] = chunkOut[i];
        prevEnd = ctxEnd;
        start = end;
        if (progress) *progress = float(start) / float(T);
    }
    return true;
}

} // namespace voxforge
