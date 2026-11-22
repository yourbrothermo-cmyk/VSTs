#include "voxforge/Notes.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace voxforge {

namespace {
float median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}
} // namespace

std::vector<Note> segmentNotes(const std::vector<float>& f0, const SegmenterConfig& cfg) {
    std::vector<Note> notes;
    const int n = int(f0.size());
    const int half = std::max(1, int(msToFrames(cfg.smoothMs, cfg.frameRate) / 2));
    const int hold = std::max(1, int(std::lround(msToFrames(cfg.splitHoldMs, cfg.frameRate))));

    for (int a = 0; a < n;) {
        if (f0[a] <= 0.0f) { ++a; continue; }
        int b = a;
        while (b < n && f0[b] > 0.0f) ++b;

        // Running median over ~one vibrato cycle: removes vibrato but keeps
        // note steps sharp (a moving average would smear them over the window).
        std::vector<float> midi(b - a), smooth(b - a), win;
        for (int t = a; t < b; ++t) midi[t - a] = hzToMidi(f0[t]);
        for (int i = 0; i < int(midi.size()); ++i) {
            const int lo = std::max(0, i - half), hi = std::min(int(midi.size()), i + half + 1);
            win.assign(midi.begin() + lo, midi.begin() + hi);
            smooth[i] = median(win);
        }

        int s = 0;
        double refSum = smooth[0];
        int refCount = 1;
        int departStart = -1;
        auto emit = [&](int from, int to) {
            if (to <= from) return;
            Note note;
            note.start = a + from;
            note.end = a + to;
            note.pitch = median(std::vector<float>(midi.begin() + from, midi.begin() + to));
            notes.push_back(note);
        };
        for (int i = 1; i < int(smooth.size()); ++i) {
            const float ref = float(refSum / refCount);
            if (std::fabs(smooth[i] - ref) > cfg.splitSemitones) {
                if (departStart < 0) departStart = i;
                if (i - departStart + 1 >= hold) {
                    // Split at the steepest point of the transition.
                    int cut = departStart;
                    float best = 0.0f;
                    for (int k = std::max(s + 1, departStart - hold); k <= departStart; ++k) {
                        const float slope = std::fabs(smooth[k] - smooth[k - 1]);
                        if (slope > best) { best = slope; cut = k; }
                    }
                    emit(s, cut);
                    s = cut;
                    refSum = 0.0;
                    refCount = 0;
                    for (int k = cut; k <= i; ++k) { refSum += smooth[k]; ++refCount; }
                    departStart = -1;
                }
            } else {
                departStart = -1;
                refSum += smooth[i];
                ++refCount;
            }
        }
        emit(s, int(smooth.size()));
        a = b;
    }

    // Merge short glide fragments (pitch between both neighbours, no gap) into
    // the closer neighbour; genuine fast notes outside that range are kept.
    const int glide = std::max(1, int(msToFrames(cfg.glideMergeMs, cfg.frameRate)));
    for (size_t i = 1; i + 1 < notes.size();) {
        Note& p = notes[i - 1]; Note& c = notes[i]; Note& nx = notes[i + 1];
        const bool contiguous = p.end == c.start && c.end == nx.start;
        const bool between = (c.pitch - p.pitch) * (nx.pitch - c.pitch) > 0.0f;
        if (contiguous && between && c.length() <= glide) {
            if (std::fabs(c.pitch - p.pitch) < std::fabs(c.pitch - nx.pitch)) p.end = c.end;
            else nx.start = c.start;
            notes.erase(notes.begin() + long(i));
        } else {
            ++i;
        }
    }
    return notes;
}

uint16_t scaleMask(int root, ScaleType scale) {
    static const std::array<std::vector<int>, int(ScaleType::Count)> intervals = {{
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        {0, 2, 4, 5, 7, 9, 11},
        {0, 2, 3, 5, 7, 8, 10},
        {0, 2, 3, 5, 7, 8, 11},
        {0, 2, 4, 7, 9},
        {0, 3, 5, 7, 10},
        {0, 3, 5, 6, 7, 10},
        {0, 2, 3, 5, 7, 9, 10},
        {0, 2, 4, 5, 7, 9, 10},
    }};
    uint16_t mask = 0;
    for (int iv : intervals[int(scale)]) mask |= uint16_t(1u << ((root + iv) % 12));
    return mask;
}

const char* scaleName(ScaleType s) {
    static const char* names[] = {"Chromatic", "Major", "Minor", "Harmonic Minor", "Major Pentatonic",
                                  "Minor Pentatonic", "Blues", "Dorian", "Mixolydian"};
    return names[int(s)];
}

std::string keyName(const Key& key) {
    static const char* roots[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    if (key.scale == ScaleType::Chromatic) return "Chromatic";
    return std::string(roots[((key.root % 12) + 12) % 12]) + " " + scaleName(key.scale);
}

Key detectKey(const std::vector<Note>& notes) {
    static const float major[12] = {6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f, 2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f};
    static const float minor[12] = {6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f, 2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f};

    // Singers often sit consistently sharp or flat; estimate that global tuning
    // offset (circular mean of per-note deviation) and remove it so it cannot
    // tilt the pitch-class histogram towards the wrong key.
    double sx = 0.0, sy = 0.0;
    for (const auto& n : notes) {
        const double dev = n.pitch - std::round(n.pitch);
        sx += n.length() * std::cos(2.0 * M_PI * dev);
        sy += n.length() * std::sin(2.0 * M_PI * dev);
    }
    const float tuning = (sx == 0.0 && sy == 0.0) ? 0.0f : float(std::atan2(sy, sx) / (2.0 * M_PI));

    std::array<double, 12> hist{};
    for (const auto& n : notes) {
        if (n.length() <= 0) continue;
        // Soft assignment: a note sung between two semitones votes for both.
        const float pitch = n.pitch - tuning;
        const float lo = std::floor(pitch);
        const float frac = pitch - lo;
        const double w = n.length();
        hist[((int(lo) % 12) + 12) % 12] += w * (1.0 - frac);
        hist[((int(lo) + 1) % 12 + 12) % 12] += w * frac;
    }
    Key best;
    best.confidence = -2.0f;
    auto correlate = [&](const float* profile, int root) {
        double mx = 0, my = 0;
        for (int i = 0; i < 12; ++i) { mx += hist[i]; my += profile[i]; }
        mx /= 12; my /= 12;
        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < 12; ++i) {
            const double dx = hist[(i + root) % 12] - mx, dy = profile[i] - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }
        return (sxx > 0 && syy > 0) ? float(sxy / std::sqrt(sxx * syy)) : 0.0f;
    };
    for (int r = 0; r < 12; ++r) {
        const float cM = correlate(major, r), cm = correlate(minor, r);
        if (cM > best.confidence) best = {r, ScaleType::Major, cM};
        if (cm > best.confidence) best = {r, ScaleType::NaturalMinor, cm};
    }
    if (best.confidence < 0.0f) best = {0, ScaleType::Chromatic, 0.0f};
    return best;
}

} // namespace voxforge
