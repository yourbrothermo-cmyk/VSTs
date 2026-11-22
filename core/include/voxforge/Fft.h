#pragma once
#include <complex>
#include <vector>

namespace voxforge {

// Small iterative radix-2 complex FFT. Kept dependency-free so the core builds
// and tests anywhere; the plugin could swap in vDSP/Accelerate later.
class Fft {
public:
    explicit Fft(int size);
    int size() const { return n_; }
    void forward(std::complex<float>* data) const { transform(data, false); }
    // Unnormalised inverse (caller divides by size()).
    void inverse(std::complex<float>* data) const { transform(data, true); }

private:
    void transform(std::complex<float>* data, bool inverse) const;
    int n_;
    std::vector<std::complex<float>> twiddles_;
    std::vector<int> bitrev_;
};

} // namespace voxforge
