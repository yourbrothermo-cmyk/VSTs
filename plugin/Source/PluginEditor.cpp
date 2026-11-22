#include "PluginEditor.h"

using namespace voxforge;

namespace {
const juce::Colour kBg(0xff15171c), kPanel(0xff1e2128), kAccent(0xff4fd1c5), kText(0xffd8dde6), kDim(0xff7a8394);
}

void PitchView::refresh() {
    auto s = proc.studio.snapshot();
    if (s && s->generation == shownGeneration) return;
    snap = s;
    shownGeneration = s ? s->generation : 0;
    repaint();
}

void PitchView::paint(juce::Graphics& g) {
    auto r = getLocalBounds().toFloat();
    g.setColour(kPanel);
    g.fillRoundedRectangle(r, 6.0f);

    if (proc.activeMode() != VoxForgeProcessor::Studio) {
        const float in = proc.liveInHz.load(), out = proc.liveOutHz.load();
        g.setColour(kText);
        g.setFont(20.0f);
        auto txt = [](float hz) {
            if (hz <= 0) return juce::String("--");
            static const char* n[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
            const float m = hzToMidi(hz);
            const int k = int(std::lround(m));
            return juce::String(n[(k % 12 + 12) % 12]) + juce::String(k / 12 - 1) + "  " +
                   juce::String(int(std::lround((m - k) * 100))) + " ct";
        };
        g.drawText("Live   in: " + txt(in) + "    out: " + txt(out), r, juce::Justification::centred);
        return;
    }
    if (!snap || !snap->analysis) {
        g.setColour(kDim);
        g.drawText("No take yet: press Capture and play the vocal through once.", r, juce::Justification::centred);
        return;
    }
    const auto& f0 = snap->analysis->pitch.f0;
    const auto& out = snap->correction.f0;
    if (f0.empty()) return;

    float lo = 127, hi = 0;
    for (float f : f0) if (f > 0) { lo = std::min(lo, hzToMidi(f)); hi = std::max(hi, hzToMidi(f)); }
    if (lo > hi) return;
    lo = std::floor(lo) - 2; hi = std::ceil(hi) + 2;
    const float w = r.getWidth(), h = r.getHeight();
    auto X = [&](float t) { return r.getX() + w * t / float(f0.size()); };
    auto Y = [&](float m) { return r.getBottom() - h * (m - lo) / (hi - lo); };

    // Semitone grid; scale tones slightly brighter.
    const uint16_t mask = scaleMask(snap->correction.key.root, snap->correction.key.scale);
    for (int m = int(lo); m <= int(hi); ++m) {
        g.setColour(inScale(mask, m) ? juce::Colour(0x22ffffff) : juce::Colour(0x0cffffff));
        g.drawHorizontalLine(int(Y(float(m))), r.getX(), r.getRight());
    }
    // Note targets.
    for (const auto& n : snap->correction.notes) {
        g.setColour(kAccent.withAlpha(0.18f * n.strength + 0.05f));
        g.fillRoundedRectangle(X(float(n.start)), Y(n.target) - 4, X(float(n.end)) - X(float(n.start)), 8, 3);
    }
    auto curve = [&](const std::vector<float>& v, juce::Colour c, float thick) {
        juce::Path p;
        bool open = false;
        for (size_t t = 0; t < v.size(); ++t) {
            if (v[t] <= 0) { open = false; continue; }
            const float x = X(float(t)), y = Y(hzToMidi(v[t]));
            if (!open) { p.startNewSubPath(x, y); open = true; } else p.lineTo(x, y);
        }
        g.setColour(c);
        g.strokePath(p, juce::PathStrokeType(thick));
    };
    curve(f0, kDim, 1.2f);
    curve(out, kAccent, 1.8f);
    g.setColour(kText);
    g.drawText(juce::String(keyName(snap->correction.key)), r.reduced(8), juce::Justification::topRight);
}

VoxForgeEditor::VoxForgeEditor(VoxForgeProcessor& p) : AudioProcessorEditor(&p), proc(p), pitchView(p) {
    setResizable(true, true);
    setResizeLimits(760, 480, 2000, 1200);
    setSize(900, 560);
    addAndMakeVisible(pitchView);

    auto& s = proc.apvts;
    auto combo = [&](juce::ComboBox& b, const char* id, std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>& a) {
        b.addItemList(s.getParameter(id)->getAllValueStrings(), 1);
        addAndMakeVisible(b);
        a = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(s, id, b);
    };
    combo(modeBox, "mode", modeAtt);
    combo(resynthBox, "resynth", resynthAtt);
    combo(rootBox, "root", rootAtt);
    combo(scaleBox, "scale", scaleAtt);
    addAndMakeVisible(autoKey);
    autoKeyAtt = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(s, "autokey", autoKey);

    captureBtn.setClickingTogglesState(true);
    captureBtn.onClick = [this] { proc.studio.setCapturing(captureBtn.getToggleState()); };
    clearBtn.onClick = [this] { proc.studio.clear(); captureBtn.setToggleState(false, juce::dontSendNotification); };
    modelBtn.onClick = [this] {
        chooser = std::make_unique<juce::FileChooser>("Choose an NSF-HiFiGAN ONNX model", juce::File(), "*.onnx");
        chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                             [this](const juce::FileChooser& fc) {
                                 const auto f = fc.getResult();
                                 if (!f.existsAsFile()) return;
                                 juce::String err;
                                 if (!proc.loadModel(f, err))
                                     juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                                                                            "Could not load model", err);
                             });
    };
    for (auto* b : {&captureBtn, &clearBtn, &modelBtn}) addAndMakeVisible(b);
    for (auto* l : {&statusLabel, &modelLabel, &latencyLabel}) {
        l->setColour(juce::Label::textColourId, kDim);
        addAndMakeVisible(l);
    }

    addKnob(amount, "amount", "Amount");
    addKnob(retune, "retune", "Retune");
    addKnob(vibrato, "vibrato", "Vibrato");
    addKnob(drift, "drift", "Drift");
    addKnob(context, "context", "Context");
    addKnob(chromatic, "chromatic", "Accidentals");
    addKnob(ornament, "ornament", "Ornaments");
    addKnob(transpose, "transpose", "Transpose");
    addKnob(formant, "formant", "Formant");
    addKnob(lowest, "lowest", "Lowest (Live)");
    addKnob(mix, "mix", "Mix");

    startTimerHz(15);
    timerCallback();
}

void VoxForgeEditor::addKnob(Knob& k, const juce::String& id, const juce::String& name) {
    k.slider.setColour(juce::Slider::rotarySliderFillColourId, kAccent);
    k.slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 18);
    k.label.setText(name, juce::dontSendNotification);
    k.label.setJustificationType(juce::Justification::centred);
    k.label.setColour(juce::Label::textColourId, kText);
    addAndMakeVisible(k.slider);
    addAndMakeVisible(k.label);
    k.att = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(proc.apvts, id, k.slider);
}

void VoxForgeEditor::timerCallback() {
    pitchView.refresh();
    if (proc.activeMode() != VoxForgeProcessor::Studio) pitchView.repaint();
    const bool studioMode = proc.activeMode() == VoxForgeProcessor::Studio;
    captureBtn.setEnabled(studioMode);
    clearBtn.setEnabled(studioMode);
    captureBtn.setToggleState(proc.studio.isCapturing(), juce::dontSendNotification);
    statusLabel.setText(studioMode ? proc.studio.status() : juce::String("Live correction active"), juce::dontSendNotification);
    modelLabel.setText("Vocoder: " + proc.modelDescription(), juce::dontSendNotification);
    const double ms = 1000.0 * proc.getLatencySamples() / std::max(1.0, proc.getSampleRate());
    latencyLabel.setText("Latency: " + juce::String(ms, 1) + " ms", juce::dontSendNotification);
    rootBox.setEnabled(!autoKey.getToggleState());
    scaleBox.setEnabled(!autoKey.getToggleState());
}

void VoxForgeEditor::paint(juce::Graphics& g) {
    g.fillAll(kBg);
    g.setColour(kText);
    g.setFont(juce::FontOptions(18.0f, juce::Font::bold));
    g.drawText("VoxForge", 14, 8, 200, 28, juce::Justification::centredLeft);
}

void VoxForgeEditor::resized() {
    auto r = getLocalBounds().reduced(12);
    auto top = r.removeFromTop(30);
    top.removeFromLeft(110);
    modeBox.setBounds(top.removeFromLeft(170).reduced(2));
    resynthBox.setBounds(top.removeFromLeft(140).reduced(2));
    autoKey.setBounds(top.removeFromLeft(90).reduced(2));
    rootBox.setBounds(top.removeFromLeft(64).reduced(2));
    scaleBox.setBounds(top.removeFromLeft(150).reduced(2));
    modelBtn.setBounds(top.removeFromRight(110).reduced(2));

    auto bar = r.removeFromTop(30);
    captureBtn.setBounds(bar.removeFromLeft(90).reduced(2));
    clearBtn.setBounds(bar.removeFromLeft(90).reduced(2));
    latencyLabel.setBounds(bar.removeFromRight(130));
    modelLabel.setBounds(bar.removeFromRight(260));
    statusLabel.setBounds(bar);

    auto knobs = r.removeFromBottom(120);
    std::vector<Knob*> ks{&amount, &retune, &vibrato, &drift, &context, &chromatic, &ornament, &transpose, &formant, &lowest, &mix};
    const int kw = knobs.getWidth() / int(ks.size());
    for (auto* k : ks) {
        auto c = knobs.removeFromLeft(kw);
        k->label.setBounds(c.removeFromTop(18));
        k->slider.setBounds(c);
    }
    pitchView.setBounds(r.reduced(0, 8));
}

juce::AudioProcessorEditor* VoxForgeProcessor::createEditor() { return new VoxForgeEditor(*this); }
