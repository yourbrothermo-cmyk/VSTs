#pragma once
#include <algorithm>
#include <atomic>
#include <vector>

namespace voxforge {

// Wait-free single-producer / single-consumer sample FIFO.
class SpscFifo {
public:
    explicit SpscFifo(size_t capacityPow2 = 1 << 17) : buf_(capacityPow2), mask_(capacityPow2 - 1) {}
    size_t available() const { return w_.load(std::memory_order_acquire) - r_.load(std::memory_order_acquire); }
    size_t space() const { return buf_.size() - available(); }
    size_t push(const float* x, size_t n) {
        const size_t w = w_.load(std::memory_order_relaxed);
        n = std::min(n, buf_.size() - (w - r_.load(std::memory_order_acquire)));
        for (size_t i = 0; i < n; ++i) buf_[(w + i) & mask_] = x[i];
        w_.store(w + n, std::memory_order_release);
        return n;
    }
    size_t pop(float* x, size_t n) {
        const size_t r = r_.load(std::memory_order_relaxed);
        n = std::min(n, w_.load(std::memory_order_acquire) - r);
        for (size_t i = 0; i < n; ++i) x[i] = buf_[(r + i) & mask_];
        r_.store(r + n, std::memory_order_release);
        return n;
    }
    // Only safe while neither side is running.
    void clear() { r_.store(0); w_.store(0); }

private:
    std::vector<float> buf_;
    size_t mask_;
    std::atomic<size_t> w_{0}, r_{0};
};

} // namespace voxforge
