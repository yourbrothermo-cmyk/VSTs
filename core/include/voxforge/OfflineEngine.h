#pragma once
#include "voxforge/MelExtractor.h"
#include "voxforge/Notes.h"
#include "voxforge/PitchCorrector.h"
#include "voxforge/PitchTracker.h"
#include "voxforge/Vocoder.h"
#include <memory>
#include <string>

namespace voxforge {

// Everything derived from the raw take that does not depend on user settings.
// Computed once per capture; correction and rendering reuse it.
struct Analysis {
    double sourceRate = kModelSampleRate;
    size_t sourceLength = 0;
    std::vector<float> audio;   // take at kModelSampleRate
    PitchTrack pitch;
    MelSpectrogram mel;
    std::vector<Note> notes;
    Key detectedKey;
};

enum class ResynthMode : int {
    NeuralHybrid = 0, // neural where pitch changed, original elsewhere (consonants, breaths, untouched notes)
    NeuralFull,       // every frame through the vocoder
    Dsp,              // TD-PSOLA, no model needed
};

struct RenderParams {
    ResynthMode mode = ResynthMode::NeuralHybrid;
    float formantSemitones = 0.0f;   // independent formant shift (neural modes)
    float hybridThresholdCents = 8.0f;
};

struct RenderResult {
    std::vector<float> audio;        // at the source sample rate, same length as the take
    std::string vocoderUsed;
    std::string error;
    bool ok = false;
};

class OfflineEngine {
public:
    Analysis analyze(const float* x, size_t n, double sampleRate) const;
    static CorrectionResult correct(const Analysis& a, const CorrectionParams& p);
    // `vocoder` may be null (falls back to DSP). Cancellable, reports progress 0..1.
    static RenderResult render(const Analysis& a, const CorrectionResult& c, const RenderParams& rp, Vocoder* vocoder,
                               const std::atomic<bool>* cancel = nullptr, float* progress = nullptr);
};

} // namespace voxforge
