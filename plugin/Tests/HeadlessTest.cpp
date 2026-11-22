// Drives the real VoxForgeProcessor the way a DAW would (fake playhead, block
// processing) to test the Studio capture -> render -> timeline playback loop,
// state save/restore, and Live mode, without a GUI or a host.
#include <JuceHeader.h>
#include "../Source/PluginProcessor.h"
#include "voxforge/OfflineEngine.h"
#include <cstdio>

using namespace voxforge;

namespace {
int failures = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("  FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

struct FakePlayHead : juce::AudioPlayHead {
    bool playing = false;
    int64_t pos = 0;
    juce::Optional<PositionInfo> getPosition() const override {
        PositionInfo p;
        p.setIsPlaying(playing);
        p.setTimeInSamples(pos);
        return p;
    }
};

std::vector<float> phrase(double sr) {
    const float notes[] = {60.35f, 62.3f, 64.45f, 65.2f, 67.4f};
    std::vector<float> x;
    double ph = 0;
    for (float m : notes) {
        for (int i = 0; i < int(0.5 * sr); ++i) {
            const double t = i / sr;
            const double f = midiToHz(m + 0.3f * float(std::sin(2 * M_PI * 5.5 * t) * std::min(1.0, t / 0.2)));
            double s = 0;
            for (int k = 1; k <= 12; ++k) if (k * f < sr / 2) s += std::sin(k * ph) / k;
            x.push_back(float(0.15 * s));
            ph += 2 * M_PI * f / sr;
        }
    }
    return x;
}

std::vector<float> run(VoxForgeProcessor& p, FakePlayHead& ph, const std::vector<float>& in, int64_t start, bool playing) {
    std::vector<float> out(in.size());
    juce::AudioBuffer<float> buf(2, 512);
    juce::MidiBuffer midi;
    ph.playing = playing;
    for (size_t i = 0; i < in.size(); i += 512) {
        const int n = int(std::min<size_t>(512, in.size() - i));
        buf.setSize(2, n, false, false, true);
        for (int c = 0; c < 2; ++c) buf.copyFrom(c, 0, in.data() + i, n);
        ph.pos = start + int64_t(i);
        p.processBlock(buf, midi);
        std::copy(buf.getReadPointer(0), buf.getReadPointer(0) + n, out.begin() + long(i));
    }
    return out;
}

bool waitReady(VoxForgeProcessor& p, int ms = 30000) {
    for (int t = 0; t < ms; t += 50) {
        if (p.studio.status().startsWith("Ready")) return true;
        juce::Thread::sleep(50);
    }
    std::printf("  status: %s\n", p.studio.status().toRawUTF8());
    return false;
}

std::vector<float> notePitches(const std::vector<float>& x, double sr) {
    auto a = OfflineEngine().analyze(x.data(), x.size(), sr);
    std::vector<float> v;
    for (auto& n : a.notes) if (n.length() > 10) v.push_back(n.pitch);
    return v;
}
} // namespace

int main() {
    juce::ScopedJuceInitialiser_GUI init;
    const double sr = 48000.0;
    const int64_t start = 96000; // take starts 2 s into the timeline
    const auto take = phrase(sr);
    const float expect[] = {60, 62, 64, 65, 67};

    std::printf("[studio] capture -> render -> playback\n");
    juce::MemoryBlock saved;
    {
        VoxForgeProcessor p;
        FakePlayHead ph;
        p.setPlayHead(&ph);
        p.prepareToPlay(sr, 512);
        EXPECT(p.getLatencySamples() == 0);

        p.studio.setCapturing(true);
        auto dryOut = run(p, ph, take, start, true);
        EXPECT(std::equal(dryOut.begin(), dryOut.end(), take.begin(), [](float a, float b) { return std::fabs(a - b) < 1e-6f; }));
        p.studio.setCapturing(false);
        p.studio.requestRender(p.correctionParams(), p.renderParams());
        EXPECT(waitReady(p));
        std::printf("  %s\n", p.studio.status().toRawUTF8());

        // Playback from a different block alignment, starting mid-timeline.
        std::vector<float> silence(take.size() + 48000, 0.0f);
        auto out = run(p, ph, silence, start - 24000, true);
        std::vector<float> region(out.begin() + 24000, out.begin() + 24000 + long(take.size()));
        auto pitches = notePitches(region, sr);
        EXPECT(pitches.size() == 5);
        for (size_t i = 0; i < std::min<size_t>(5, pitches.size()); ++i) {
            std::printf("  note %zu: %.3f (target %.0f)\n", i, pitches[i], expect[i]);
            EXPECT(std::fabs(pitches[i] - expect[i]) < 0.1f);
        }
        // Outside the take, input passes through.
        EXPECT(std::all_of(out.begin(), out.begin() + 23000, [](float v) { return v == 0.0f; }));

        // Transport stopped: plugin monitors the input.
        auto stopped = run(p, ph, take, start, false);
        EXPECT(std::fabs(stopped[1000] - take[1000]) < 1e-6f);

        p.getStateInformation(saved);
    }

    std::printf("[state] restore session from saved state\n");
    {
        VoxForgeProcessor p;
        FakePlayHead ph;
        p.setPlayHead(&ph);
        p.setStateInformation(saved.getData(), int(saved.getSize())); // before prepare, like many hosts
        p.prepareToPlay(sr, 512);
        p.studio.requestRender(p.correctionParams(), p.renderParams());
        EXPECT(waitReady(p));
        std::vector<float> silence(take.size(), 0.0f);
        auto out = run(p, ph, silence, start, true);
        auto pitches = notePitches(out, sr);
        EXPECT(pitches.size() == 5);
        for (size_t i = 0; i < std::min<size_t>(5, pitches.size()); ++i) EXPECT(std::fabs(pitches[i] - expect[i]) < 0.1f);
    }

    std::printf("[live] DSP live mode\n");
    {
        VoxForgeProcessor p;
        FakePlayHead ph;
        p.setPlayHead(&ph);
        auto* modeParam = p.apvts.getParameter("mode");
        modeParam->setValueNotifyingHost(modeParam->convertTo0to1(1.0f));
        auto* autokey = p.apvts.getParameter("autokey");
        autokey->setValueNotifyingHost(0.0f); // C major
        p.prepareToPlay(sr, 512);
        EXPECT(p.activeMode() == VoxForgeProcessor::Live);
        const int lat = p.getLatencySamples();
        std::printf("  reported latency: %d samples (%.1f ms)\n", lat, 1000.0 * lat / sr);
        EXPECT(lat > 0 && lat < int(0.05 * sr));
        std::vector<float> held(take.begin() + 24000 * 2 + 4000, take.begin() + 24000 * 2 + 20000); // inside note 3 (64.45)
        std::vector<float> loop;
        for (int r = 0; r < 6; ++r) loop.insert(loop.end(), held.begin(), held.end());
        auto out = run(p, ph, loop, 0, true);
        std::vector<float> tail(out.begin() + 20000, out.end());
        auto pitches = notePitches(tail, sr);
        EXPECT(!pitches.empty());
        if (!pitches.empty()) std::printf("  live output pitch %.3f (target 64)\n", pitches[0]);
        if (!pitches.empty()) EXPECT(std::fabs(pitches[0] - 64.0f) < 0.15f);
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
