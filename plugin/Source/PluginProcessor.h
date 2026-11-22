#pragma once
#include <JuceHeader.h>
#include "StudioSession.h"
#include "voxforge/LiveEngine.h"

class VoxForgeProcessor : public juce::AudioProcessor, private juce::Timer {
public:
    enum Mode { Studio = 0, Live = 1, LiveHQ = 2 };

    VoxForgeProcessor();
    ~VoxForgeProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Message thread.
    bool loadModel(const juce::File& file, juce::String& error);
    juce::String modelDescription() const;
    voxforge::CorrectionParams correctionParams() const;
    voxforge::RenderParams renderParams() const;
    Mode activeMode() const { return Mode(mode.load()); }

    juce::AudioProcessorValueTreeState apvts;
    StudioSession studio;
    std::atomic<float> liveInHz{0.0f}, liveOutHz{0.0f};

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

private:
    void timerCallback() override;
    void applyMode(Mode m);

    double sampleRate = 44100.0;
    int blockSize = 512;
    std::atomic<int> mode{Studio};
    voxforge::LiveDspEngine liveDsp;
    voxforge::LiveNeuralEngine liveNeural;
    std::shared_ptr<voxforge::Vocoder> vocoder;
    juce::File modelFile;
    juce::AudioBuffer<float> mono;
    juce::AudioBuffer<float> dry;
    voxforge::CorrectionParams lastLiveParams;
    std::atomic<bool> liveParamsDirty{true};
    juce::String lastParamHash;
    float lastLowestNote = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VoxForgeProcessor)
};
