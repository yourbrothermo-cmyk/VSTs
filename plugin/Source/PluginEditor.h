#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

// Draws the analysed take: sung pitch (grey), corrected pitch (accent) and
// the note targets chosen by the context-aware corrector.
class PitchView : public juce::Component {
public:
    explicit PitchView(VoxForgeProcessor& p) : proc(p) {}
    void paint(juce::Graphics& g) override;
    void refresh();

private:
    VoxForgeProcessor& proc;
    std::shared_ptr<const StudioSession::Snapshot> snap;
    uint32_t shownGeneration = 0;
};

class VoxForgeEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit VoxForgeEditor(VoxForgeProcessor&);
    ~VoxForgeEditor() override = default;
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    struct Knob {
        juce::Slider slider{juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow};
        juce::Label label;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> att;
    };
    void addKnob(Knob& k, const juce::String& id, const juce::String& name);

    VoxForgeProcessor& proc;
    PitchView pitchView;
    juce::ComboBox modeBox, resynthBox, rootBox, scaleBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> modeAtt, resynthAtt, rootAtt, scaleAtt;
    juce::ToggleButton autoKey{"Auto key"};
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> autoKeyAtt;
    juce::TextButton captureBtn{"Capture"}, clearBtn{"Clear take"}, modelBtn{"Load model..."};
    juce::Label statusLabel, modelLabel, latencyLabel;
    Knob amount, retune, vibrato, drift, context, chromatic, ornament, transpose, formant, lowest, mix;
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VoxForgeEditor)
};
