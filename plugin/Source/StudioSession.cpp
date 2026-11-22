#include "StudioSession.h"

using namespace voxforge;

StudioSession::StudioSession() : juce::Thread("VoxForge Studio") { startThread(juce::Thread::Priority::low); }

StudioSession::~StudioSession() {
    cancelRender.store(true);
    stopThread(10000);
    delete current.exchange(nullptr);
}

void StudioSession::prepare(double sampleRate, double maxSeconds) {
    if (sampleRate != hostRate && hasTake()) {
        // A take captured at another rate cannot be replayed sample-aligned.
        clear();
        setStatus("Sample rate changed: take cleared, please capture again.");
    }
    hostRate = sampleRate;
    const size_t want = size_t(maxSeconds * sampleRate);
    if (captureBuf.size() != want) captureBuf.assign(want, 0.0f);
    if (pendingLoad.isValid()) {
        auto t = pendingLoad;
        pendingLoad = {};
        loadState(t);
    }
}

void StudioSession::setVocoder(std::shared_ptr<Vocoder> v) {
    {
        const juce::ScopedLock sl(lock);
        vocoder = std::move(v);
    }
    if (analysis) {
        needRender.store(true);
        notify();
    }
}

void StudioSession::process(const float* in, float* out, int n, bool playing, int64_t pos) {
    if (capturing.load() && playing && pos >= 0) {
        int64_t o = origin.load();
        if (o < 0) { o = pos; origin.store(o); }
        int64_t end = capturedEnd.load();
        for (int i = 0; i < n; ++i) {
            const int64_t idx = pos + i - o;
            if (idx >= 0 && idx < int64_t(captureBuf.size())) {
                captureBuf[size_t(idx)] = in[i];
                end = std::max(end, idx + 1);
            }
        }
        capturedEnd.store(end);
    }

    // Safe hand-off of the rendered buffer (see publishRender()).
    Render* r;
    do {
        r = current.load();
        audioUsing.store(r);
    } while (r != current.load());

    if (r && playing && pos >= 0 && !capturing.load()) {
        const int64_t len = int64_t(r->audio.size());
        for (int i = 0; i < n; ++i) {
            const int64_t idx = pos + i - r->origin;
            out[i] = (idx >= 0 && idx < len) ? r->audio[size_t(idx)] : in[i];
        }
    } else if (out != in) {
        std::copy(in, in + n, out);
    }
    audioUsing.store(nullptr);
}

void StudioSession::publishRender(Render* r) {
    Render* old = current.exchange(r);
    while (old && audioUsing.load() == old) juce::Thread::sleep(1);
    delete old;
}

void StudioSession::setCapturing(bool shouldCapture) {
    if (shouldCapture && !capturing.load()) {
        cancelRender.store(true);
        publishRender(nullptr); // monitor the dry input while recording
        setStatus("Capturing... play the vocal section, then stop the transport.");
    }
    capturing.store(shouldCapture);
    if (!shouldCapture && hasTake()) requestAnalysis();
}

void StudioSession::requestAnalysis() {
    needAnalysis.store(true);
    cancelRender.store(true);
    notify();
}

void StudioSession::requestRender(const CorrectionParams& cp, const RenderParams& rp) {
    {
        const juce::ScopedLock sl(lock);
        pendingCp = cp;
        pendingRp = rp;
    }
    if (!capturing.load() && hasTake()) {
        cancelRender.store(true);
        needRender.store(true);
        notify();
    }
}

void StudioSession::clear() {
    capturing.store(false);
    cancelRender.store(true);
    publishRender(nullptr);
    origin.store(-1);
    capturedEnd.store(0);
    std::fill(captureBuf.begin(), captureBuf.end(), 0.0f);
    const juce::ScopedLock sl(lock);
    analysis.reset();
    snap.reset();
    takeFile = juce::File();
    statusText = "Arm Capture, then play the vocal through once.";
}

std::shared_ptr<const StudioSession::Snapshot> StudioSession::snapshot() const {
    const juce::ScopedLock sl(lock);
    return snap;
}

juce::String StudioSession::status() const {
    const juce::ScopedLock sl(lock);
    return statusText;
}

void StudioSession::setStatus(const juce::String& s) {
    const juce::ScopedLock sl(lock);
    statusText = s;
}

void StudioSession::run() {
    while (!threadShouldExit()) {
        wait(200);
        if (threadShouldExit()) break;

        if (needAnalysis.exchange(false) && !capturing.load()) {
            const int64_t len = capturedEnd.load();
            if (len <= 0) continue;
            setStatus("Analysing take...");
            auto a = std::make_shared<Analysis>(OfflineEngine().analyze(captureBuf.data(), size_t(len), hostRate));
            {
                const juce::ScopedLock sl(lock);
                analysis = a;
            }
            setStatus("Analysed: " + juce::String(int(a->notes.size())) + " notes, key " +
                      juce::String(keyName(a->detectedKey)));
            needRender.store(true);
        }

        if (needRender.exchange(false) && !capturing.load()) {
            std::shared_ptr<const Analysis> a;
            std::shared_ptr<Vocoder> voc;
            CorrectionParams cp;
            RenderParams rp;
            {
                const juce::ScopedLock sl(lock);
                a = analysis;
                voc = vocoder;
                cp = pendingCp;
                rp = pendingRp;
            }
            if (!a) continue;
            juce::Thread::sleep(150); // debounce knob drags
            if (needRender.load() || needAnalysis.load()) continue;
            cancelRender.store(false);

            auto snapNew = std::make_shared<Snapshot>();
            snapNew->analysis = a;
            snapNew->correction = OfflineEngine::correct(*a, cp);
            snapNew->origin = origin.load();
            snapNew->hostRate = hostRate;
            snapNew->generation = ++generation;
            {
                const juce::ScopedLock sl(lock);
                snap = snapNew; // UI shows the new curve immediately
            }

            setStatus("Rendering (" + juce::String(voc && rp.mode != ResynthMode::Dsp ? voc->name() : "TD-PSOLA") + ")...");
            renderProgress.store(0.0f);
            float progress = 0.0f;
            const RenderResult rr = OfflineEngine::render(*a, snapNew->correction, rp, voc.get(), &cancelRender, &progress);
            renderProgress.store(1.0f);
            if (!rr.ok) {
                if (!cancelRender.load()) setStatus("Render failed: " + juce::String(rr.error));
                continue;
            }
            auto* r = new Render();
            r->origin = snapNew->origin;
            r->audio = rr.audio;
            publishRender(r);
            setStatus("Ready: " + juce::String(keyName(snapNew->correction.key)) + " | " + juce::String(rr.vocoderUsed) +
                      (rr.error.empty() ? juce::String() : " (" + juce::String(rr.error) + ")"));
        }
    }
}

juce::ValueTree StudioSession::saveState() {
    juce::ValueTree t("Studio");
    const int64_t len = capturedEnd.load();
    if (len <= 0 || capturing.load()) return t;
    if (!takeFile.existsAsFile()) {
        auto dir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                       .getChildFile("VoxForge").getChildFile("Takes");
        dir.createDirectory();
        takeFile = dir.getChildFile(juce::Uuid().toString() + ".wav");
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> os(takeFile.createOutputStream());
        if (os) {
            std::unique_ptr<juce::AudioFormatWriter> w(wav.createWriterFor(os.get(), hostRate, 1, 32, {}, 0));
            if (w) {
                os.release();
                const float* ch = captureBuf.data();
                w->writeFromFloatArrays(&ch, 1, int(len));
            }
        }
    }
    t.setProperty("file", takeFile.getFullPathName(), nullptr);
    t.setProperty("origin", juce::int64(origin.load()), nullptr);
    return t;
}

void StudioSession::loadState(const juce::ValueTree& t) {
    if (captureBuf.empty()) { pendingLoad = t.createCopy(); return; }
    const juce::File f(t.getProperty("file").toString());
    if (!f.existsAsFile()) return;
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatReader> r(wav.createReaderFor(f.createInputStream().release(), true));
    if (!r) return;
    if (std::abs(r->sampleRate - hostRate) > 1.0) {
        setStatus("Saved take was recorded at a different sample rate; please capture again.");
        return;
    }
    const int64_t len = std::min<int64_t>(r->lengthInSamples, int64_t(captureBuf.size()));
    juce::AudioBuffer<float> buf(1, int(len));
    r->read(&buf, 0, int(len), 0, true, false);
    std::copy(buf.getReadPointer(0), buf.getReadPointer(0) + len, captureBuf.begin());
    origin.store(juce::int64(t.getProperty("origin")));
    capturedEnd.store(len);
    takeFile = f;
    requestAnalysis();
}
