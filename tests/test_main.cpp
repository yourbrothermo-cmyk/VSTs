#include "Signals.h"
#include "TestUtil.h"
#include "voxforge/Fft.h"
#include "voxforge/LiveEngine.h"
#include "voxforge/MelExtractor.h"
#include "voxforge/Notes.h"
#include "voxforge/OfflineEngine.h"
#include "voxforge/PitchCorrector.h"
#include "voxforge/PitchTracker.h"
#include "voxforge/Resampler.h"
#include "voxforge/Vocoder.h"
#include <algorithm>
#include <chrono>
#include <complex>
#include <numeric>
#include <thread>

using namespace voxforge;

namespace {
std::vector<float> sine(double f, double sr, size_t n, float a = 0.5f) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = a * float(std::sin(2 * M_PI * f * i / sr));
    return x;
}
float voicedMedian(const std::vector<float>& f0, size_t from, size_t to) {
    std::vector<float> v;
    for (size_t i = from; i < std::min(to, f0.size()); ++i) if (f0[i] > 0) v.push_back(f0[i]);
    return medianOf(v);
}
} // namespace

TEST(fft_matches_dft) {
    const int N = 64;
    Fft fft(N);
    std::vector<std::complex<float>> x(N), y;
    for (int i = 0; i < N; ++i) x[i] = {float(std::sin(i * 0.3) + 0.2 * i), float(std::cos(i * 0.7))};
    y = x;
    fft.forward(y.data());
    for (int k = 0; k < N; ++k) {
        std::complex<double> acc = 0;
        for (int n = 0; n < N; ++n) acc += std::complex<double>(x[n]) * std::polar(1.0, -2 * M_PI * k * n / N);
        CHECK_NEAR(y[k].real(), acc.real(), 1e-3);
        CHECK_NEAR(y[k].imag(), acc.imag(), 1e-3);
    }
}

TEST(pitch_tracks_sine_and_harmonic_tone) {
    PitchTracker tr;
    auto x = sine(220.0, kModelSampleRate, kModelSampleRate);
    auto p = tr.analyze(x.data(), x.size());
    CHECK(p.size() == x.size() / kHopSize);
    CHECK_NEAR(voicedMedian(p.f0, 5, p.size() - 5), 220.0, 0.5);

    auto h = harmonicTone(std::vector<float>(kModelSampleRate, 110.0f), kModelSampleRate);
    p = tr.analyze(h.data(), h.size());
    CHECK_NEAR(voicedMedian(p.f0, 5, p.size() - 5), 110.0, 0.5);
    int voiced = 0;
    for (float f : p.f0) voiced += f > 0;
    CHECK(voiced > int(p.size()) - 6);
}

TEST(pitch_is_silent_on_silence) {
    PitchTracker tr;
    std::vector<float> z(20000, 0.0f);
    auto p = tr.analyze(z.data(), z.size());
    for (float f : p.f0) CHECK(f == 0.0f);
}

TEST(mel_shape_and_peak) {
    MelExtractor mel;
    auto x = sine(1000.0, kModelSampleRate, 22050);
    auto m = mel.compute(x.data(), x.size());
    CHECK(m.numFrames == int(x.size() / kHopSize));
    const float* f = m.frame(m.numFrames / 2);
    const int peak = int(std::max_element(f, f + kNumMels) - f);
    CHECK_NEAR(mel.binCentresHz()[peak], 1000.0, 60.0);
    for (float v : m.data) CHECK(std::isfinite(v));

    MelSpectrogram w = m;
    warpFormants(w, 12.0f);
    const float* g = w.frame(w.numFrames / 2);
    const int peak2 = int(std::max_element(g, g + kNumMels) - g);
    CHECK_NEAR(mel.binCentresHz()[peak2], 2000.0, 150.0);
}

TEST(resamplers_preserve_frequency_and_level) {
    auto x = sine(1000.0, 48000.0, 48000);
    auto y = resample(x.data(), x.size(), 48000.0, 44100.0);
    CHECK(y.size() == 44100);
    float peak = 0;
    for (size_t i = 1000; i < y.size() - 1000; ++i) peak = std::max(peak, std::fabs(y[i]));
    CHECK_NEAR(peak, 0.5, 0.01);
    PitchTracker tr;
    auto p = tr.analyze(y.data(), y.size());
    CHECK_NEAR(voicedMedian(p.f0, 5, p.size() - 5), 1000.0, 2.0);

    StreamingResampler rs(48000.0, 44100.0);
    std::vector<float> out(50000);
    int total = 0;
    for (size_t i = 0; i < x.size(); i += 256)
        total += rs.process(x.data() + i, int(std::min<size_t>(256, x.size() - i)), out.data() + total, int(out.size()) - total);
    CHECK(std::abs(total - 44100) < 64);
    peak = 0;
    for (int i = 1000; i < total - 1000; ++i) peak = std::max(peak, std::fabs(out[size_t(i)]));
    CHECK_NEAR(peak, 0.5, 0.01);
}

TEST(segmenter_keeps_vibrato_inside_notes) {
    auto f = noteContour({60.2f, 62.1f, 64.3f}, 0.6, kFrameRate, 0.5f); // per-frame contour
    auto notes = segmentNotes(f);
    CHECK(notes.size() == 3);
    if (notes.size() == 3) {
        CHECK_NEAR(notes[0].pitch, 60.2, 0.1);
        CHECK_NEAR(notes[1].pitch, 62.1, 0.1);
        CHECK_NEAR(notes[2].pitch, 64.3, 0.1);
    }
}

TEST(key_detection_reference_free) {
    std::vector<Note> notes;
    const int melody[] = {60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 67, 60};
    int t = 0;
    for (int m : melody) { notes.push_back({t, t + 30, float(m) + 0.1f}); t += 32; }
    Key k = detectKey(notes);
    CHECK(k.root == 0 && k.scale == ScaleType::Major);

    // Same melody sung ~40 cents sharp throughout must still read as C major.
    for (auto& n : notes) n.pitch += 0.3f;
    k = detectKey(notes);
    CHECK(k.root == 0 && k.scale == ScaleType::Major);
}

TEST(context_keeps_intervals_of_flat_phrase) {
    // C-D-E sung 55 cents flat. Note-by-note snapping picks B for the first note;
    // context keeps the whole-tone shape and lands on C-D-E.
    std::vector<Note> notes = {{0, 40, 59.45f}, {42, 82, 61.45f}, {84, 124, 63.45f}};
    CorrectionParams p;
    auto a = notes;
    chooseTargets(a, scaleMask(0, ScaleType::Major), p);
    CHECK(a[0].target == 60 && a[1].target == 62 && a[2].target == 64);
    p.contextStrength = 0.0f;
    auto b = notes;
    chooseTargets(b, scaleMask(0, ScaleType::Major), p);
    CHECK(b[0].target == 59);
}

TEST(ornaments_are_corrected_less) {
    std::vector<Note> notes = {{0, 3, 60.4f}, {3, 60, 62.3f}};
    CorrectionParams p;
    chooseTargets(notes, scaleMask(0, ScaleType::Major), p);
    CHECK(notes[0].strength < 0.6f);
    CHECK_NEAR(notes[1].strength, 1.0, 1e-6);
}

TEST(correction_preserves_vibrato) {
    // One note 30 cents sharp of A4 with +/-40 cent, 5.5 Hz vibrato.
    auto f0 = noteContour({69.3f}, 2.0, kFrameRate, 0.4f);
    auto notes = segmentNotes(f0);
    CorrectionParams p;
    p.autoKey = false;
    p.key = {0, ScaleType::Chromatic, 1.0f};
    p.driftCorrection = 1.0f;
    auto r = correctPitch(f0, notes, p);
    const size_t a = 40, b = f0.size() - 40;
    double meanIn = 0, meanOut = 0;
    for (size_t t = a; t < b; ++t) { meanIn += hzToMidi(f0[t]); meanOut += hzToMidi(r.f0[t]); }
    meanIn /= double(b - a); meanOut /= double(b - a);
    CHECK_NEAR(meanOut, 69.0, 0.03);
    double vIn = 0, vOut = 0;
    for (size_t t = a; t < b; ++t) {
        vIn += std::pow(hzToMidi(f0[t]) - meanIn, 2);
        vOut += std::pow(hzToMidi(r.f0[t]) - meanOut, 2);
    }
    CHECK_NEAR(std::sqrt(vOut / vIn), 1.0, 0.08); // vibrato depth kept

    p.vibratoDepth = 0.0f;
    auto flat = correctPitch(f0, notes, p);
    double vFlat = 0;
    for (size_t t = a; t < b; ++t) vFlat += std::pow(hzToMidi(flat.f0[t]) - 69.0, 2);
    CHECK(std::sqrt(vFlat / vIn) < 0.15);
}

TEST(psola_shifts_pitch) {
    const double sr = kModelSampleRate;
    auto contour = std::vector<float>(size_t(sr), 200.0f);
    auto x = harmonicTone(contour, sr);
    PitchTracker tr;
    auto p = tr.analyze(x.data(), x.size());
    std::vector<float> dst = p.f0;
    for (auto& f : dst) if (f > 0) f *= 1.25f;
    auto y = psolaShift(x.data(), x.size(), sr, kHopSize, p.f0, dst);
    auto q = tr.analyze(y.data(), y.size());
    CHECK_NEAR(voicedMedian(q.f0, 10, q.size() - 10), 250.0, 2.0);
}

TEST(offline_engine_corrects_a_phrase_dsp) {
    const double sr = 48000.0;
    auto contour = noteContour({60.35f, 62.3f, 64.4f, 65.25f}, 0.5, sr, 0.25f);
    auto x = harmonicTone(contour, sr);
    OfflineEngine eng;
    auto a = eng.analyze(x.data(), x.size(), sr);
    CHECK(a.notes.size() == 4);
    CorrectionParams cp;
    cp.autoKey = false;
    cp.key = {0, ScaleType::Major, 1.0f};
    auto c = OfflineEngine::correct(a, cp);
    RenderParams rp;
    rp.mode = ResynthMode::Dsp;
    auto r = OfflineEngine::render(a, c, rp, nullptr);
    CHECK(r.ok && r.audio.size() == x.size());
    auto b = eng.analyze(r.audio.data(), r.audio.size(), sr);
    const float expect[] = {60, 62, 64, 65};
    CHECK(b.notes.size() == 4);
    for (size_t i = 0; i < std::min<size_t>(4, b.notes.size()); ++i) CHECK_NEAR(b.notes[i].pitch, expect[i], 0.12);
}

TEST(live_dsp_engine_snaps_in_real_time) {
    const double sr = 48000.0;
    LiveDspEngine eng;
    eng.prepare(sr, 80.0f);
    CorrectionParams p;
    p.key = {0, ScaleType::Major, 1.0f};
    p.retuneMs = 20.0f;
    eng.setParams(p);
    CHECK(eng.latencySamples() > 0 && eng.latencySamples() < int(0.06 * sr));
    auto x = harmonicTone(std::vector<float>(size_t(sr * 1.5), midiToHz(55.4f)), sr); // G3 + 40 cents
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); i += 256) eng.process(x.data() + i, y.data() + i, int(std::min<size_t>(256, x.size() - i)));
    std::vector<float> tail(y.begin() + long(sr * 0.5), y.end());
    auto t44 = resample(tail.data(), tail.size(), sr, kModelSampleRate);
    PitchTracker tr;
    auto q = tr.analyze(t44.data(), t44.size());
    CHECK_NEAR(hzToMidi(voicedMedian(q.f0, 5, q.size() - 5)), 55.0, 0.08);
}

namespace {
// Fake vocoder: emits a DC level of f0/1000 per frame, so the test can check
// pipeline alignment, chunking and FIFO hand-off without a model.
class DcVocoder : public Vocoder {
public:
    std::string name() const override { return "dc"; }
    bool synthesize(const float*, int T, const float* f0, std::vector<float>& out, std::string&) override {
        out.resize(size_t(T) * kHopSize);
        for (int t = 0; t < T; ++t)
            std::fill(out.begin() + long(t) * kHopSize, out.begin() + long(t + 1) * kHopSize, f0[t] / 1000.0f);
        return true;
    }
};
} // namespace

TEST(live_neural_engine_pipeline) {
    const double sr = 48000.0;
    LiveNeuralEngine eng;
    CorrectionParams p;
    p.key = {0, ScaleType::Major, 1.0f};
    p.retuneMs = 0.0f;
    eng.setParams(p);
    eng.prepare(sr, 512, std::make_shared<DcVocoder>());
    CHECK(eng.latencySamples() > 0 && eng.latencySamples() < int(0.15 * sr));
    auto x = harmonicTone(std::vector<float>(size_t(sr * 2), midiToHz(69.3f)), sr);
    std::vector<float> y(x.size());
    // Pace blocks at real time, as a host would; the worker must keep up.
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < x.size(); i += 512) {
        eng.process(x.data() + i, y.data() + i, int(std::min<size_t>(512, x.size() - i)));
        std::this_thread::sleep_until(t0 + std::chrono::microseconds(long((i + 512) * 1e6 / sr)));
    }
    eng.stop();
    CHECK(eng.underruns() == 0);
    // Steady state should read 440 Hz / 1000 (A4 corrected from +30 cents).
    std::vector<float> tail(y.begin() + long(sr * 0.8), y.end());
    CHECK_NEAR(medianOf(tail), 0.440, 0.004);
}

int main() {
    for (auto& t : registry()) {
        const int before = failures();
        t.fn();
        std::printf("%s %s\n", failures() == before ? "[ ok ]" : "[FAIL]", t.name.c_str());
    }
    std::printf("\n%d failure(s)\n", failures());
    return failures() == 0 ? 0 : 1;
}
