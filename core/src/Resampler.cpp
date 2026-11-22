#include "voxforge/Resampler.h"
#include <algorithm>
#include <cmath>

namespace voxforge {

namespace {
double besselI0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 30; ++k) {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
    }
    return sum;
}
double kaiser(double x, double halfWidth, double beta = 8.0) {
    const double r = x / halfWidth;
    if (std::fabs(r) >= 1.0) return 0.0;
    return besselI0(beta * std::sqrt(1.0 - r * r)) / besselI0(beta);
}
double sinc(double x) { return std::fabs(x) < 1e-9 ? 1.0 : std::sin(M_PI * x) / (M_PI * x); }
} // namespace

std::vector<float> resample(const float* in, size_t n, double inRate, double outRate) {
    if (std::fabs(inRate - outRate) < 1e-6) return std::vector<float>(in, in + n);
    const double ratio = outRate / inRate;
    const double cutoff = std::min(1.0, ratio) * 0.95;
    const int half = int(std::ceil(24.0 / cutoff));
    const size_t outN = size_t(std::floor(n * ratio));
    std::vector<float> out(outN);
    for (size_t i = 0; i < outN; ++i) {
        const double centre = i / ratio;
        const long c = long(std::floor(centre));
        double acc = 0.0;
        for (long k = c - half + 1; k <= c + half; ++k) {
            if (k < 0 || size_t(k) >= n) continue;
            const double d = centre - k;
            acc += in[k] * cutoff * sinc(cutoff * d) * kaiser(d, half);
        }
        out[i] = float(acc);
    }
    return out;
}

StreamingResampler::StreamingResampler(double inRate, double outRate, int halfTaps)
    : step_(inRate / outRate), cutoff_(std::min(1.0, outRate / inRate) * 0.95), halfTaps_(halfTaps) {
    hist_.assign(4096, 0.0f);
    const int len = halfTaps_ * kTableRes;
    table_.resize(size_t(len) + 2);
    for (int i = 0; i <= len + 1; ++i) {
        const double x = double(i) / kTableRes;
        table_[i] = float(cutoff_ * sinc(cutoff_ * x) * kaiser(x, halfTaps_));
    }
    reset();
}

void StreamingResampler::reset() {
    std::fill(hist_.begin(), hist_.end(), 0.0f);
    written_ = 0;
    pos_ = 0.0;
}

float StreamingResampler::tap(double x) const {
    const double p = std::fabs(x) * kTableRes; // kernel is symmetric
    const size_t i = size_t(p);
    if (i + 1 >= table_.size()) return 0.0f;
    const float fr = float(p - double(i));
    return table_[i] + (table_[i + 1] - table_[i]) * fr;
}

int StreamingResampler::process(const float* in, int numIn, float* out, int maxOut) {
    const size_t mask = hist_.size() - 1;
    int produced = 0;
    for (int i = 0; i < numIn; ++i) {
        hist_[size_t(written_) & mask] = in[i];
        ++written_;
        // Emit every output whose kernel is now fully covered by received input.
        while (produced < maxOut && pos_ + halfTaps_ < double(written_)) {
            const long long c = (long long)std::floor(pos_);
            double acc = 0.0;
            for (long long k = c - halfTaps_ + 1; k <= c + halfTaps_; ++k) {
                if (k < 0) continue;
                acc += hist_[size_t(k) & mask] * tap(pos_ - double(k));
            }
            out[produced++] = float(acc);
            pos_ += step_;
        }
    }
    return produced;
}

} // namespace voxforge
