#pragma once
#include <cstddef>
#include <vector>

namespace voxforge {

// Offline band-limited (Kaiser-windowed sinc) sample-rate conversion.
std::vector<float> resample(const float* in, size_t n, double inRate, double outRate);

// Streaming fixed-ratio resampler for the live path. Allocation-free after
// construction. Group delay is latencyInputSamples() at the input rate.
class StreamingResampler {
public:
    StreamingResampler(double inRate, double outRate, int halfTaps = 16);
    void reset();
    // Consumes all of `in`, writes up to maxOut samples, returns count written.
    int process(const float* in, int numIn, float* out, int maxOut);
    int latencyInputSamples() const { return halfTaps_; }
    double ratio() const { return step_; }

private:
    double step_;      // input samples advanced per output sample
    double cutoff_;
    int halfTaps_;
    std::vector<float> hist_; // ring of recent input samples
    std::vector<float> table_; // kernel sampled at kTableRes points per input sample
    static constexpr int kTableRes = 512;
    long long written_ = 0;   // total input samples received
    double pos_ = 0.0;        // next output position in input-sample time
    float tap(double x) const;
};

} // namespace voxforge
