#pragma once
#include "voxforge/Types.h"
#include <algorithm>
#include <cmath>
#include <vector>

// Voice-like test signal: 10 harmonics with 1/k rolloff following an f0 contour (Hz per sample).
inline std::vector<float> harmonicTone(const std::vector<float>& f0PerSample, double sr, float amp = 0.3f) {
    std::vector<float> x(f0PerSample.size());
    double phase = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double f = f0PerSample[i];
        double s = 0.0;
        if (f > 0) for (int k = 1; k <= 10; ++k) if (k * f < sr / 2) s += std::sin(k * phase) / k;
        x[i] = float(amp * s);
        phase += 2.0 * M_PI * f / sr;
        if (phase > 2.0 * M_PI) phase -= 2.0 * M_PI;
    }
    return x;
}

// Sequence of notes (fractional MIDI) with vibrato, as an f0-per-sample contour.
inline std::vector<float> noteContour(const std::vector<float>& midi, double noteSec, double sr, float vibSemis = 0.0f,
                                      float vibHz = 5.5f) {
    std::vector<float> f;
    for (float m : midi) {
        const size_t n = size_t(noteSec * sr);
        for (size_t i = 0; i < n; ++i) {
            const double t = i / sr;
            f.push_back(voxforge::midiToHz(m + vibSemis * float(std::sin(2 * M_PI * vibHz * t))));
        }
    }
    return f;
}

inline float medianOf(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}
