// Offline command-line front-end to the VoxForge core: the same pipeline as
// the plugin's Studio mode. Useful for tuning, regression tests and batch work.
#include "Wav.h"
#include "voxforge/OfflineEngine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

using namespace voxforge;

static void usage() {
    std::puts(
        "usage: voxforge_cli in.wav out.wav [options]\n"
        "  --model PATH        NSF-HiFiGAN .onnx (omit for DSP fallback)\n"
        "  --mode hybrid|full|dsp\n"
        "  --key auto|C:major|F#:minor|A:chromatic ...\n"
        "  --amount 0..1  --retune MS  --vibrato 0..2  --drift 0..1  --context 0..2\n"
        "  --transpose SEMIS  --formant SEMIS\n"
        "  --dump PREFIX       write PREFIX.mel.f32 [T,128], PREFIX.f0.f32, PREFIX.f0c.f32 (44.1k audio in PREFIX.44k.wav)");
}

static bool parseKey(const std::string& s, Key& k) {
    static const std::map<std::string, int> roots = {{"C", 0}, {"C#", 1}, {"Db", 1}, {"D", 2}, {"D#", 3}, {"Eb", 3},
        {"E", 4}, {"F", 5}, {"F#", 6}, {"Gb", 6}, {"G", 7}, {"G#", 8}, {"Ab", 8}, {"A", 9}, {"A#", 10}, {"Bb", 10}, {"B", 11}};
    const auto c = s.find(':');
    if (c == std::string::npos || !roots.count(s.substr(0, c))) return false;
    k.root = roots.at(s.substr(0, c));
    for (int i = 0; i < int(ScaleType::Count); ++i) {
        std::string n = scaleName(ScaleType(i));
        for (auto& ch : n) ch = char(std::tolower(ch));
        if (n == s.substr(c + 1)) { k.scale = ScaleType(i); return true; }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }
    std::string model, dump, mode = "hybrid";
    CorrectionParams cp;
    RenderParams rp;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--mode") mode = next();
        else if (a == "--key") { const auto v = next(); if (v != "auto") { cp.autoKey = false; if (!parseKey(v, cp.key)) { std::fprintf(stderr, "bad key %s\n", v.c_str()); return 2; } } }
        else if (a == "--amount") cp.amount = std::stof(next());
        else if (a == "--retune") cp.retuneMs = std::stof(next());
        else if (a == "--vibrato") cp.vibratoDepth = std::stof(next());
        else if (a == "--drift") cp.driftCorrection = std::stof(next());
        else if (a == "--context") cp.contextStrength = std::stof(next());
        else if (a == "--transpose") cp.transposeSemitones = std::stof(next());
        else if (a == "--formant") rp.formantSemitones = std::stof(next());
        else if (a == "--dump") dump = next();
        else { usage(); return 2; }
    }
    rp.mode = mode == "full" ? ResynthMode::NeuralFull : mode == "dsp" ? ResynthMode::Dsp : ResynthMode::NeuralHybrid;

    WavData in;
    std::string err;
    if (!readWav(argv[1], in, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    OfflineEngine eng;
    const Analysis an = eng.analyze(in.mono.data(), in.mono.size(), in.sampleRate);
    const CorrectionResult cr = OfflineEngine::correct(an, cp);
    std::printf("frames %zu, notes %zu, key %s (detected %s, r=%.2f)\n", an.pitch.size(), an.notes.size(),
                keyName(cr.key).c_str(), keyName(an.detectedKey).c_str(), an.detectedKey.confidence);
    for (const auto& n : cr.notes)
        std::printf("  %7.3fs  sung %6.2f -> %3d  (%+4.0f cents, strength %.2f)\n", n.start / kFrameRate, n.pitch,
                    int(n.target), (n.target - n.pitch) * 100.0f, n.strength);

    if (!dump.empty()) {
        auto w = [&](const std::string& p, const std::vector<float>& v) {
            FILE* f = std::fopen(p.c_str(), "wb"); std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); };
        w(dump + ".mel.f32", an.mel.data);
        w(dump + ".f0.f32", an.pitch.f0);
        w(dump + ".f0c.f32", cr.f0);
        writeWav(dump + ".44k.wav", an.audio, kModelSampleRate);
    }

    std::unique_ptr<Vocoder> voc;
    if (!model.empty() && rp.mode != ResynthMode::Dsp) {
        voc = loadNsfHifiGan(model, err);
        if (!voc) std::fprintf(stderr, "model: %s (falling back to DSP)\n", err.c_str());
    }
    float progress = 0;
    const RenderResult rr = OfflineEngine::render(an, cr, rp, voc.get(), nullptr, &progress);
    if (!rr.ok) { std::fprintf(stderr, "render failed: %s\n", rr.error.c_str()); return 1; }
    if (!rr.error.empty()) std::fprintf(stderr, "note: %s\n", rr.error.c_str());
    writeWav(argv[2], rr.audio, in.sampleRate);
    std::printf("rendered with %s -> %s\n", rr.vocoderUsed.c_str(), argv[2]);
    return 0;
}
