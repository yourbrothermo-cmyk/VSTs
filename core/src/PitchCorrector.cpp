#include "voxforge/PitchCorrector.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace voxforge {

namespace {
constexpr int kCandRadius = 2; // candidates: round(pitch) +/- 2 semitones

float durationMs(const Note& n, double frameRate) { return float(n.length() * 1000.0 / frameRate); }

float unaryCost(const Note& n, int m, uint16_t scale, const CorrectionParams& p) {
    const float dev = n.pitch - float(m);
    float cost = 4.0f * dev * dev; // 1 semitone away == 4
    if (!inScale(scale, m)) {
        float penalty = 1.5f;
        // A long note sung right on a chromatic pitch is probably intended.
        if (std::fabs(dev) * 100.0f < p.chromaticToleranceCents && durationMs(n, p.frameRate) > 150.0f)
            penalty *= 0.1f;
        cost += penalty;
    }
    return cost;
}

float pairCost(const Note& a, int ma, const Note& b, int mb, const CorrectionParams& p) {
    const float gapMs = float((b.start - a.end) * 1000.0 / p.frameRate);
    if (gapMs > 600.0f) return 0.0f; // phrases separated by a rest are independent
    const float sung = b.pitch - a.pitch;
    const float cand = float(mb - ma);
    const float d = sung - cand;
    float cost = 2.0f * p.contextStrength * d * d;
    // Rarely intended leaps (tritone, major 7th) need extra evidence.
    const int iv = std::abs(mb - ma) % 12;
    if ((iv == 6 || iv == 11) && std::fabs(sung - cand) > 0.25f) cost += 0.3f * p.contextStrength;
    return cost;
}
} // namespace

void chooseTargets(std::vector<Note>& notes, uint16_t scale, const CorrectionParams& p) {
    const int N = int(notes.size());
    if (N == 0) return;
    const int C = 2 * kCandRadius + 1;
    std::vector<float> cost(size_t(N) * C);
    std::vector<int> back(size_t(N) * C, 0);
    auto cand = [&](int i, int c) { return int(std::lround(notes[i].pitch)) - kCandRadius + c; };

    for (int c = 0; c < C; ++c) cost[c] = unaryCost(notes[0], cand(0, c), scale, p);
    for (int i = 1; i < N; ++i) {
        for (int c = 0; c < C; ++c) {
            const int m = cand(i, c);
            float best = std::numeric_limits<float>::max();
            int arg = 0;
            for (int pc = 0; pc < C; ++pc) {
                const float v = cost[size_t(i - 1) * C + pc] + pairCost(notes[i - 1], cand(i - 1, pc), notes[i], m, p);
                if (v < best) { best = v; arg = pc; }
            }
            cost[size_t(i) * C + c] = best + unaryCost(notes[i], m, scale, p);
            back[size_t(i) * C + c] = arg;
        }
    }
    int c = int(std::min_element(cost.end() - C, cost.end()) - (cost.end() - C));
    for (int i = N - 1; i >= 0; --i) {
        notes[i].target = float(cand(i, c));
        c = back[size_t(i) * C + c];
    }
    for (auto& n : notes) {
        const float ms = durationMs(n, p.frameRate);
        n.strength = p.ornamentMs > 0.0f ? std::clamp(ms / p.ornamentMs, 0.0f, 1.0f) : 1.0f;
    }
}

CorrectionResult correctPitch(const std::vector<float>& f0, std::vector<Note> notes, const CorrectionParams& p) {
    CorrectionResult r;
    const int n = int(f0.size());
    r.key = p.autoKey ? detectKey(notes) : p.key;
    chooseTargets(notes, scaleMask(r.key.root, r.key.scale), p);

    std::vector<float> midi(n, 0.0f), centre(n, 0.0f), shift(n, 0.0f), resid(n, 0.0f);
    std::vector<char> hasShift(n, 0);
    for (int t = 0; t < n; ++t) midi[t] = hzToMidi(f0[t]);

    const int half = std::max(1, int(msToFrames(p.vibratoWindowMs, p.frameRate) / 2));
    for (const auto& note : notes) {
        // Centre line: moving average inside the note, window shrinking at edges.
        for (int t = note.start; t < note.end; ++t) {
            const int lo = std::max(note.start, t - half), hi = std::min(note.end, t + half + 1);
            double acc = 0.0;
            for (int k = lo; k < hi; ++k) acc += midi[k];
            centre[t] = float(acc / (hi - lo));
        }
        for (int t = note.start; t < note.end; ++t) {
            const float anchor = p.driftCorrection * centre[t] + (1.0f - p.driftCorrection) * note.pitch;
            shift[t] = p.amount * note.strength * (note.target - anchor);
            resid[t] = midi[t] - centre[t];
            hasShift[t] = 1;
        }
    }

    // Fill frames without a note-derived shift (unvoiced gaps) by holding the
    // nearest value, so smoothing only glides between notes, never into onsets.
    int last = -1;
    for (int t = 0; t < n; ++t) {
        if (hasShift[t]) { last = t; continue; }
        int next = t;
        while (next < n && !hasShift[next]) ++next;
        for (int k = t; k < next; ++k) {
            if (last >= 0 && next < n) shift[k] = (k - last <= next - k) ? shift[last] : shift[next];
            else if (last >= 0) shift[k] = shift[last];
            else if (next < n) shift[k] = shift[next];
        }
        t = next - 1;
    }

    // Retune speed: zero-phase one-pole smoothing of the shift curve.
    if (p.retuneMs > 0.0f && n > 1) {
        const float a = std::exp(-1.0f / float(std::max(1e-3, msToFrames(p.retuneMs, p.frameRate))));
        for (int t = 1; t < n; ++t) shift[t] = a * shift[t - 1] + (1.0f - a) * shift[t];
        for (int t = n - 2; t >= 0; --t) shift[t] = a * shift[t + 1] + (1.0f - a) * shift[t];
    }

    r.f0.assign(n, 0.0f);
    r.shiftSemis.assign(n, 0.0f);
    for (int t = 0; t < n; ++t) {
        if (f0[t] <= 0.0f) continue;
        const float s = shift[t] + (p.vibratoDepth - 1.0f) * resid[t] + p.transposeSemitones;
        r.shiftSemis[t] = s;
        r.f0[t] = midiToHz(midi[t] + s);
    }
    r.notes = std::move(notes);
    return r;
}

void LiveCorrector::prepare(double frameRate) {
    frameRate_ = frameRate;
    setParams(p_);
    reset();
}

void LiveCorrector::setParams(const CorrectionParams& p) {
    p_ = p;
    mask_ = scaleMask(p.key.root, p.key.scale);
    // ~ vibratoWindow time constant for the causal centre line.
    centreCoef_ = std::exp(-1.0f / float(std::max(1.0, msToFrames(p.vibratoWindowMs * 0.5, frameRate_))));
    retuneCoef_ = p.retuneMs > 0.0f ? std::exp(-1.0f / float(std::max(1e-3, msToFrames(p.retuneMs, frameRate_)))) : 0.0f;
}

void LiveCorrector::reset() {
    centre_ = 0.0f;
    target_ = -1;
    shift_ = 0.0f;
    wasVoiced_ = false;
}

float LiveCorrector::process(float f0) {
    if (f0 <= 0.0f) {
        wasVoiced_ = false;
        return 0.0f;
    }
    const float m = hzToMidi(f0);
    if (!wasVoiced_) {
        centre_ = m;
        target_ = -1;
    } else {
        centre_ = centreCoef_ * centre_ + (1.0f - centreCoef_) * m;
    }

    // Nearest allowed note with hysteresis so vibrato near a boundary does not flip targets.
    auto nearestAllowed = [&](float x) {
        int best = int(std::lround(x));
        float bestD = 1e9f;
        for (int k = int(std::floor(x)) - 6; k <= int(std::ceil(x)) + 6; ++k) {
            if (!inScale(mask_, k)) continue;
            const float d = std::fabs(x - k);
            if (d < bestD) { bestD = d; best = k; }
        }
        return best;
    };
    const int cand = nearestAllowed(centre_);
    if (target_ < 0 || (cand != target_ && std::fabs(centre_ - target_) > std::fabs(centre_ - cand) + 0.15f))
        target_ = cand;

    const float desired = p_.amount * (target_ - centre_);
    shift_ = wasVoiced_ ? retuneCoef_ * shift_ + (1.0f - retuneCoef_) * desired : desired;
    wasVoiced_ = true;
    const float resid = m - centre_;
    return midiToHz(m + shift_ + (p_.vibratoDepth - 1.0f) * resid + p_.transposeSemitones);
}

} // namespace voxforge
