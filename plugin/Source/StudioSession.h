#pragma once
#include <JuceHeader.h>
#include "voxforge/OfflineEngine.h"
#include <atomic>
#include <memory>

// Studio mode: the "transfer" workflow. While armed, incoming audio is captured
// against the host timeline. When the take is analysed, the whole performance
// is available, so note choice, key detection and vibrato separation can use
// full context (past and future). The corrected take is rendered on a worker
// thread and played back sample-aligned with the timeline, with zero latency.
class StudioSession : private juce::Thread {
public:
    struct Snapshot {
        std::shared_ptr<const voxforge::Analysis> analysis;
        voxforge::CorrectionResult correction;
        int64_t origin = 0;
        double hostRate = 44100.0;
        uint32_t generation = 0;
    };

    StudioSession();
    ~StudioSession() override;

    void prepare(double sampleRate, double maxSeconds = 600.0);
    void setVocoder(std::shared_ptr<voxforge::Vocoder> v);

    // Audio thread.
    void process(const float* in, float* out, int n, bool playing, int64_t timelineSample);

    // Message thread.
    void setCapturing(bool shouldCapture);
    bool isCapturing() const { return capturing.load(); }
    bool hasTake() const { return capturedEnd.load() > 0; }
    void requestAnalysis();
    void requestRender(const voxforge::CorrectionParams& cp, const voxforge::RenderParams& rp);
    void clear();
    std::shared_ptr<const Snapshot> snapshot() const;
    juce::String status() const;
    float progress() const { return renderProgress.load(); }

    // Persistence: the take is stored as a WAV next to the user's app data.
    juce::ValueTree saveState();
    void loadState(const juce::ValueTree& tree);

private:
    struct Render {
        int64_t origin = 0;
        std::vector<float> audio;
    };

    void run() override;
    void publishRender(Render* r);
    void setStatus(const juce::String& s);

    double hostRate = 44100.0;
    std::vector<float> captureBuf;
    std::atomic<bool> capturing{false};
    std::atomic<int64_t> origin{-1};
    std::atomic<int64_t> capturedEnd{0};

    std::atomic<Render*> current{nullptr};
    std::atomic<Render*> audioUsing{nullptr};

    std::shared_ptr<voxforge::Vocoder> vocoder;
    juce::CriticalSection lock; // guards vocoder, params, analysis, snapshot, status
    voxforge::CorrectionParams pendingCp;
    voxforge::RenderParams pendingRp;
    std::shared_ptr<const voxforge::Analysis> analysis;
    std::shared_ptr<const Snapshot> snap;
    juce::String statusText {"Arm Capture, then play the vocal through once."};
    juce::File takeFile;
    juce::ValueTree pendingLoad; // state restored before prepare()

    std::atomic<bool> needAnalysis{false}, needRender{false}, cancelRender{false};
    std::atomic<float> renderProgress{0.0f};
    uint32_t generation = 0;
};
