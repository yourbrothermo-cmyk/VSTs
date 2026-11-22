#include "PluginProcessor.h"
#include "PluginEditor.h"

using namespace voxforge;

namespace {
juce::StringArray scaleNames() {
    juce::StringArray s;
    for (int i = 0; i < int(ScaleType::Count); ++i) s.add(scaleName(ScaleType(i)));
    return s;
}
const juce::StringArray kRoots{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

float get(const juce::AudioProcessorValueTreeState& s, const char* id) { return s.getRawParameterValue(id)->load(); }
} // namespace

juce::AudioProcessorValueTreeState::ParameterLayout VoxForgeProcessor::createLayout() {
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout l;
    auto pct = AudioParameterFloatAttributes().withLabel("%");
    auto ms = AudioParameterFloatAttributes().withLabel("ms");
    auto st = AudioParameterFloatAttributes().withLabel("st");
    l.add(std::make_unique<AudioParameterChoice>(ParameterID{"mode", 1}, "Mode", StringArray{"Studio", "Live", "Live HQ (neural)"}, 0));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"amount", 1}, "Amount", NormalisableRange<float>(0, 100), 100, pct));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"retune", 1}, "Retune Speed", NormalisableRange<float>(0, 400, 0, 0.5f), 40, ms));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"vibrato", 1}, "Vibrato", NormalisableRange<float>(0, 200), 100, pct));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"drift", 1}, "Drift Correction", NormalisableRange<float>(0, 100), 50, pct));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"context", 1}, "Context", NormalisableRange<float>(0, 200), 100, pct));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"chromatic", 1}, "Accidental Tolerance", NormalisableRange<float>(0, 50), 20,
                                                AudioParameterFloatAttributes().withLabel("ct")));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"ornament", 1}, "Ornament Length", NormalisableRange<float>(0, 200), 70, ms));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"transpose", 1}, "Transpose", NormalisableRange<float>(-24, 24, 1), 0, st));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"formant", 1}, "Formant", NormalisableRange<float>(-12, 12, 0.1f), 0, st));
    l.add(std::make_unique<AudioParameterChoice>(ParameterID{"resynth", 1}, "Resynthesis", StringArray{"Neural Hybrid", "Neural Full", "DSP"}, 0));
    l.add(std::make_unique<AudioParameterBool>(ParameterID{"autokey", 1}, "Auto Key", true));
    l.add(std::make_unique<AudioParameterChoice>(ParameterID{"root", 1}, "Key", kRoots, 0));
    l.add(std::make_unique<AudioParameterChoice>(ParameterID{"scale", 1}, "Scale", scaleNames(), 1));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"lowest", 1}, "Lowest Note (Live)", NormalisableRange<float>(60, 200), 80,
                                                AudioParameterFloatAttributes().withLabel("Hz")));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{"mix", 1}, "Mix", NormalisableRange<float>(0, 100), 100, pct));
    return l;
}

VoxForgeProcessor::VoxForgeProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "VoxForge", createLayout()) {
    startTimerHz(15);
}

VoxForgeProcessor::~VoxForgeProcessor() {
    stopTimer();
    liveNeural.stop();
}

bool VoxForgeProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto in = layouts.getMainInputChannelSet(), out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo()) return false;
    return in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo();
}

CorrectionParams VoxForgeProcessor::correctionParams() const {
    CorrectionParams p;
    p.amount = get(apvts, "amount") / 100.0f;
    p.retuneMs = get(apvts, "retune");
    p.vibratoDepth = get(apvts, "vibrato") / 100.0f;
    p.driftCorrection = get(apvts, "drift") / 100.0f;
    p.contextStrength = get(apvts, "context") / 100.0f;
    p.chromaticToleranceCents = get(apvts, "chromatic");
    p.ornamentMs = get(apvts, "ornament");
    p.transposeSemitones = get(apvts, "transpose");
    p.autoKey = get(apvts, "autokey") > 0.5f;
    p.key.root = int(get(apvts, "root"));
    p.key.scale = ScaleType(int(get(apvts, "scale")));
    return p;
}

RenderParams VoxForgeProcessor::renderParams() const {
    RenderParams r;
    r.mode = ResynthMode(int(get(apvts, "resynth")));
    r.formantSemitones = get(apvts, "formant");
    return r;
}

void VoxForgeProcessor::prepareToPlay(double sr, int maxBlock) {
    sampleRate = sr;
    blockSize = maxBlock;
    mono.setSize(1, maxBlock);
    dry.setSize(1, maxBlock);
    studio.prepare(sr);
    lastLowestNote = get(apvts, "lowest");
    liveDsp.prepare(sr, lastLowestNote);
    liveDsp.setParams(correctionParams());
    liveNeural.prepare(sr, maxBlock, vocoder);
    liveNeural.setParams(correctionParams());
    applyMode(Mode(int(get(apvts, "mode"))));
}

void VoxForgeProcessor::releaseResources() { liveNeural.stop(); }

void VoxForgeProcessor::applyMode(Mode m) {
    if (m == LiveHQ && !vocoder) m = Live; // no model: fall back to the DSP live path
    mode.store(m);
    setLatencySamples(m == Studio ? 0 : m == Live ? liveDsp.latencySamples() : liveNeural.latencySamples());
}

void VoxForgeProcessor::timerCallback() {
    // Mode, latency and engine re-preparation happen here, off the audio thread.
    const Mode wanted = Mode(int(get(apvts, "mode")));
    const float lowest = get(apvts, "lowest");
    if (std::abs(lowest - lastLowestNote) > 0.5f && wanted == Live) {
        suspendProcessing(true);
        lastLowestNote = lowest;
        liveDsp.prepare(sampleRate, lowest);
        liveDsp.setParams(correctionParams());
        applyMode(wanted);
        suspendProcessing(false);
    }
    if (wanted != activeMode() && !(wanted == LiveHQ && !vocoder && activeMode() == Live)) {
        suspendProcessing(true);
        liveDsp.reset();
        applyMode(wanted);
        suspendProcessing(false);
    }

    // Studio: re-render when any setting changes.
    const auto cp = correctionParams();
    const auto rp = renderParams();
    const juce::String hash = juce::String(cp.amount) + "|" + juce::String(cp.retuneMs) + "|" + juce::String(cp.vibratoDepth) +
                              "|" + juce::String(cp.driftCorrection) + "|" + juce::String(cp.contextStrength) + "|" +
                              juce::String(cp.chromaticToleranceCents) + "|" + juce::String(cp.ornamentMs) + "|" +
                              juce::String(cp.transposeSemitones) + "|" + juce::String(int(cp.autoKey)) + "|" +
                              juce::String(cp.key.root) + "|" + juce::String(int(cp.key.scale)) + "|" +
                              juce::String(int(rp.mode)) + "|" + juce::String(rp.formantSemitones);
    if (hash != lastParamHash) {
        lastParamHash = hash;
        studio.requestRender(cp, rp);
        liveParamsDirty.store(true);
    }
}

void VoxForgeProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;
    const int n = buffer.getNumSamples();
    const int inCh = getTotalNumInputChannels(), outCh = getTotalNumOutputChannels();
    if (n > mono.getNumSamples()) { mono.setSize(1, n, false, false, true); dry.setSize(1, n, false, false, true); }

    // Vocals are processed mono: sum inputs, duplicate to every output.
    float* m = mono.getWritePointer(0);
    juce::FloatVectorOperations::clear(m, n);
    for (int c = 0; c < inCh; ++c) juce::FloatVectorOperations::addWithMultiply(m, buffer.getReadPointer(c), 1.0f / float(inCh), n);
    float* d = dry.getWritePointer(0);
    juce::FloatVectorOperations::copy(d, m, n);

    switch (activeMode()) {
        case Studio: {
            bool playing = false;
            int64_t pos = -1;
            if (auto* ph = getPlayHead())
                if (auto info = ph->getPosition()) {
                    playing = info->getIsPlaying();
                    if (auto t = info->getTimeInSamples()) pos = *t;
                }
            studio.process(d, m, n, playing, pos);
            break;
        }
        case Live:
            if (liveParamsDirty.exchange(false)) liveDsp.setParams(correctionParams());
            liveDsp.process(d, m, n);
            liveInHz.store(liveDsp.lastInputF0());
            liveOutHz.store(liveDsp.lastOutputF0());
            break;
        case LiveHQ:
            if (liveParamsDirty.exchange(false)) {
                liveNeural.setFormantShift(get(apvts, "formant"));
                if (!liveNeural.setParams(correctionParams())) liveParamsDirty.store(true);
            }
            liveNeural.process(d, m, n);
            break;
    }

    // Mix (Studio only: in Live modes the dry path is not latency-aligned).
    const float mix = get(apvts, "mix") / 100.0f;
    if (activeMode() == Studio && mix < 0.999f)
        for (int i = 0; i < n; ++i) m[i] = mix * m[i] + (1.0f - mix) * d[i];

    for (int c = 0; c < outCh; ++c) buffer.copyFrom(c, 0, m, n);
}

bool VoxForgeProcessor::loadModel(const juce::File& file, juce::String& error) {
    std::string err;
    std::shared_ptr<Vocoder> v(loadNsfHifiGan(file.getFullPathName().toStdString(), err));
    if (!v) {
        error = err;
        return false;
    }
    suspendProcessing(true);
    vocoder = v;
    modelFile = file;
    liveNeural.prepare(sampleRate, blockSize, vocoder);
    liveNeural.setParams(correctionParams());
    applyMode(Mode(int(get(apvts, "mode"))));
    suspendProcessing(false);
    studio.setVocoder(v);
    return true;
}

juce::String VoxForgeProcessor::modelDescription() const {
    if (vocoder) return modelFile.getFileName();
    return onnxRuntimeAvailable() ? "No model loaded (DSP fallback)" : "Built without ONNX Runtime (DSP only)";
}

void VoxForgeProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts.copyState();
    state.setProperty("model", modelFile.getFullPathName(), nullptr);
    state.removeChild(state.getChildWithName("Studio"), nullptr);
    state.appendChild(studio.saveState(), nullptr);
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void VoxForgeProcessor::setStateInformation(const void* data, int size) {
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, size));
    if (!xml) return;
    auto state = juce::ValueTree::fromXml(*xml);
    if (!state.hasType(apvts.state.getType())) return;
    const juce::File model(state.getProperty("model").toString());
    studio.loadState(state.getChildWithName("Studio"));
    state.removeChild(state.getChildWithName("Studio"), nullptr);
    apvts.replaceState(state);
    juce::String err;
    if (model.existsAsFile()) loadModel(model, err);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new VoxForgeProcessor(); }
