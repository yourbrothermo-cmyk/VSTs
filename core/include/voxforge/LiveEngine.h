#pragma once
#include "voxforge/MelExtractor.h"
#include "voxforge/PitchCorrector.h"
#include "voxforge/PitchTracker.h"
#include "voxforge/Resampler.h"
#include "voxforge/SpscFifo.h"
#include "voxforge/Vocoder.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace voxforge {

// Live mode, DSP path: streaming YIN + causal corrector + streaming TD-PSOLA.
// Runs entirely on the audio thread, allocation-free after prepare().
// Latency is ~3.25 periods of the lowest expected note (about 30-45 ms).
class LiveDspEngine {
public:
    void prepare(double sampleRate, float lowestNoteHz = 80.0f);
    void setParams(const CorrectionParams& p);
    void reset();
    void process(const float* in, float* out, int n);
    int latencySamples() const { return latency_; }
    float lastInputF0() const { return lastSrcF0_; }
    float lastOutputF0() const { return lastDstF0_; }

private:
    float f0At(const std::vector<float>& ring, double s) const;
    void analyseFrame();
    void placeMarks();
    void synthesise();

    double sr_ = 44100.0;
    std::unique_ptr<PitchTracker> tracker_;
    LiveCorrector corrector_;
    int hop_ = 256, span_ = 0, maxP_ = 0, latency_ = 0;
    size_t mask_ = 0;
    std::vector<float> in_, acc_, frameBuf_;
    std::vector<float> srcF0_, dstF0_; // per-frame rings
    size_t frameMask_ = 0;
    long long written_ = 0, framesDone_ = 0;
    std::vector<double> markPos_, markPer_;
    size_t markMask_ = 0;
    long long markCount_ = 0, nearest_ = 0;
    double nextMark_ = 0.0, nextTs_ = -1.0;
    float lastSrcF0_ = 0.0f, lastDstF0_ = 0.0f;
};

// Live mode, neural path ("Live HQ"): features are computed per hop on a worker
// thread, NSF-HiFiGAN runs on short chunks with left context, and audio is
// handed back through lock-free FIFOs. Latency is fixed and reported to the
// host; it depends on chunk size (default ~90 ms). If the machine cannot keep
// up, underruns are counted and silence is emitted (alignment is kept).
class LiveNeuralEngine {
public:
    struct Config {
        int chunkFrames = 2;    // new frames per vocoder call
        int contextFrames = 12; // left context
        int lookaheadFrames = 2;
    };
    ~LiveNeuralEngine();
    void prepare(double hostRate, int maxBlock, std::shared_ptr<Vocoder> vocoder, Config cfg);
    void prepare(double hostRate, int maxBlock, std::shared_ptr<Vocoder> vocoder) {
        prepare(hostRate, maxBlock, std::move(vocoder), Config{});
    }
    // Non-blocking; returns false if the worker holds the lock (retry next block).
    bool setParams(const CorrectionParams& p);
    void setFormantShift(float semitones) { formant_.store(semitones); }
    void stop();
    void process(const float* in, float* out, int n); // audio thread
    int latencySamples() const { return latency_; }
    long long underruns() const { return underruns_.load(); }
    bool running() const { return running_.load(); }

private:
    void workerLoop();
    bool workerStep();

    double hostRate_ = 44100.0;
    Config cfg_;
    std::shared_ptr<Vocoder> vocoder_;
    std::unique_ptr<StreamingResampler> inRs_, outRs_;
    std::unique_ptr<SpscFifo> inFifo_, outFifo_;
    std::vector<float> scratch_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<long long> underruns_{0};
    std::atomic<float> formant_{0.0f};
    int latency_ = 0, wait_ = 0;
    long long hostIn_ = 0, hostOut_ = 0, owed_ = 0;

    // Worker-side state.
    std::mutex paramMutex_;
    CorrectionParams pendingParams_;
    std::atomic<bool> paramsDirty_{false};
    LiveCorrector corrector_;
    std::unique_ptr<PitchTracker> tracker_;
    MelExtractor melEx_;
    std::vector<float> hist_;           // 44.1k input ring
    long long hist44_ = 0;
    std::vector<float> melRing_, f0Ring_; // per-frame rings
    long long nextFeat_ = 0, nextOut_ = 0;
    std::vector<float> prevTail_;
    std::vector<float> frameWin_, pitchWin_, melChunk_, f0Chunk_, vocOut_, rsOut_;
};

} // namespace voxforge
