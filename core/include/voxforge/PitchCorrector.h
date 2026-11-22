#pragma once
#include "voxforge/Notes.h"
#include "voxforge/Types.h"
#include <vector>

namespace voxforge {

struct CorrectionParams {
    float amount = 1.0f;              // 0..1: how far note centres move to their targets
    float retuneMs = 40.0f;           // glide time between corrected notes (0 = hard snap)
    float vibratoDepth = 1.0f;        // 1 = keep vibrato as sung, 0 = flatten, 2 = exaggerate
    float driftCorrection = 0.5f;     // 0 = keep slow drift inside a note, 1 = flatten it
    float contextStrength = 1.0f;     // weight of melodic-interval context in target choice
    float chromaticToleranceCents = 20.0f; // accurately sung out-of-key notes are kept if within this
    float ornamentMs = 70.0f;         // notes shorter than this are ornaments/transitions: corrected less
    float transposeSemitones = 0.0f;  // global pitch shift applied after correction
    bool autoKey = true;
    Key key;                          // used when autoKey == false
    double frameRate = kFrameRate;
    float vibratoWindowMs = 180.0f;   // window that separates note centre from vibrato
};

struct CorrectionResult {
    std::vector<float> f0;            // corrected f0 (Hz), 0 = unvoiced
    std::vector<float> shiftSemis;    // applied shift per frame (for display / hybrid blend)
    std::vector<Note> notes;          // notes with targets/strengths filled in
    Key key;
};

// Context-aware target selection: a Viterbi search over candidate semitones
// per note. Costs combine (a) distance from the sung pitch, (b) scale
// membership, softened for deliberately sung accidentals, and (c) agreement of
// the corrected melodic interval with the interval the singer actually sang,
// so a phrase sung uniformly flat keeps its shape instead of being split
// between neighbouring scale degrees.
void chooseTargets(std::vector<Note>& notes, uint16_t scale, const CorrectionParams& p);

// Full offline correction. The f0 contour is decomposed per note into a slow
// centre line and a residual (vibrato, scoops, micro-pitch); only the centre is
// moved, so vibrato, its rate and its shape are preserved.
CorrectionResult correctPitch(const std::vector<float>& f0, std::vector<Note> notes, const CorrectionParams& p);

// Causal per-frame corrector for Live mode (no look-ahead).
class LiveCorrector {
public:
    void prepare(double frameRate);
    void setParams(const CorrectionParams& p);
    void reset();
    // Returns the corrected f0 for this frame (0 stays 0).
    float process(float f0Hz);
    float currentShift() const { return shift_; }

private:
    CorrectionParams p_;
    uint16_t mask_ = 0xFFF;
    double frameRate_ = kFrameRate;
    float centre_ = 0.0f;      // one-pole low-passed MIDI (vibrato-removed)
    int target_ = -1;          // current target note (hysteresis)
    float shift_ = 0.0f;
    bool wasVoiced_ = false;
    float centreCoef_ = 0.0f, retuneCoef_ = 0.0f;
};

} // namespace voxforge
