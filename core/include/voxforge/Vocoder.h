#pragma once
#include "voxforge/Types.h"
#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace voxforge {

// Neural vocoder: (log-mel, f0) -> waveform at kModelSampleRate,
// numFrames * kHopSize samples. Implementations must be callable from a
// worker thread; they are never called from the audio thread.
class Vocoder {
public:
    virtual ~Vocoder() = default;
    virtual std::string name() const = 0;
    virtual bool synthesize(const float* mel, int numFrames, const float* f0, std::vector<float>& out,
                            std::string& error) = 0;
};

// NSF-HiFiGAN via ONNX Runtime. Returns nullptr (with `error` set) when the
// model cannot be loaded or when the core was built without ONNX Runtime.
// Accepts both OpenVPI ([1,T,128] mel + [1,T] f0) and [1,128,T] layouts.
std::unique_ptr<Vocoder> loadNsfHifiGan(const std::string& modelPath, std::string& error, int numThreads = 0);
bool onnxRuntimeAvailable();

// Long takes are vocoded in overlapping chunks (bounded memory, cancellable).
bool synthesizeChunked(Vocoder& v, const MelSpectrogram& mel, const std::vector<float>& f0, std::vector<float>& out,
                       std::string& error, const std::atomic<bool>* cancel = nullptr, float* progress = nullptr,
                       int chunkFrames = 768, int overlapFrames = 24);

// TD-PSOLA pitch shifting: the DSP fallback (no model required) and the engine
// behind Live mode. Grains are taken at analysis pitch marks and re-spaced at
// the target period, which preserves the spectral envelope (formants) for
// moderate shifts.
std::vector<float> psolaShift(const float* x, size_t n, double sampleRate, int hop,
                              const std::vector<float>& f0Src, const std::vector<float>& f0Dst);

} // namespace voxforge
