#include "voxforge/Fft.h"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace voxforge {

Fft::Fft(int size) : n_(size) {
    if (size < 2 || (size & (size - 1)) != 0)
        throw std::invalid_argument("FFT size must be a power of two");
    int bits = 0;
    while ((1 << bits) < size) ++bits;
    bitrev_.resize(size);
    for (int i = 0; i < size; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        bitrev_[i] = r;
    }
    twiddles_.resize(size / 2);
    for (int i = 0; i < size / 2; ++i) {
        const double a = -2.0 * M_PI * i / size;
        twiddles_[i] = {float(std::cos(a)), float(std::sin(a))};
    }
}

void Fft::transform(std::complex<float>* x, bool inverse) const {
    for (int i = 0; i < n_; ++i)
        if (i < bitrev_[i]) std::swap(x[i], x[bitrev_[i]]);
    for (int len = 2; len <= n_; len <<= 1) {
        const int half = len / 2;
        const int step = n_ / len;
        for (int i = 0; i < n_; i += len) {
            for (int j = 0; j < half; ++j) {
                auto w = twiddles_[j * step];
                if (inverse) w = std::conj(w);
                const auto u = x[i + j];
                const auto v = x[i + j + half] * w;
                x[i + j] = u + v;
                x[i + j + half] = u - v;
            }
        }
    }
}

} // namespace voxforge
