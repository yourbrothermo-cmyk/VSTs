#include "voxforge/LiveEngine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace voxforge {

namespace {
size_t pow2AtLeast(size_t v) { size_t p = 1; while (p < v) p <<= 1; return p; }
constexpr int kSynthEvery = 64;
} // namespace

// ---------------------------------------------------------------- LiveDspEngine

void LiveDspEngine::prepare(double sampleRate, float lowestNoteHz) {
    sr_ = sampleRate;
    hop_ = std::max(64, int(std::lround(sampleRate * 256.0 / 44100.0)));
    PitchTrackerConfig c;
    c.sampleRate = sampleRate;
    c.hop = hop_;
    c.integrationWindow = 768;
    c.fmin = lowestNoteHz;
    c.fmax = 1100.0f;
    tracker_ = std::make_unique<PitchTracker>(c);
    span_ = tracker_->frameSpan();
    maxP_ = int(std::ceil(sampleRate / lowestNoteHz));
    latency_ = int(std::ceil(3.25 * maxP_)) + kSynthEvery + 16;

    const size_t ring = pow2AtLeast(size_t(std::max(1 << 16, 4 * (span_ + latency_))));
    in_.assign(ring, 0.0f);
    acc_.assign(ring, 0.0f);
    mask_ = ring - 1;
    frameBuf_.assign(size_t(span_), 0.0f);
    srcF0_.assign(1024, 0.0f);
    dstF0_.assign(1024, 0.0f);
    frameMask_ = 1023;
    markPos_.assign(4096, 0.0);
    markPer_.assign(4096, 0.0);
    markMask_ = 4095;
    corrector_.prepare(sampleRate / hop_);
    reset();
}

void LiveDspEngine::setParams(const CorrectionParams& p) {
    CorrectionParams q = p;
    q.frameRate = sr_ / hop_;
    corrector_.setParams(q);
}

void LiveDspEngine::reset() {
    std::fill(in_.begin(), in_.end(), 0.0f);
    std::fill(acc_.begin(), acc_.end(), 0.0f);
    written_ = framesDone_ = markCount_ = nearest_ = 0;
    nextMark_ = 0.0;
    nextTs_ = -1.0;
    corrector_.reset();
}

float LiveDspEngine::f0At(const std::vector<float>& ring, double s) const {
    if (framesDone_ == 0) return 0.0f;
    const double ft = (s - span_ * 0.5) / hop_;
    const long long last = framesDone_ - 1, first = std::max(0LL, framesDone_ - 1000);
    auto get = [&](long long t) { return ring[size_t(std::clamp(t, first, last)) & frameMask_]; };
    const long long t0 = (long long)std::floor(ft);
    const float a = get(t0), b = get(t0 + 1);
    if (a <= 0.0f || b <= 0.0f) return (ft - double(t0) < 0.5) ? a : b;
    const float fr = float(std::clamp(ft - double(t0), 0.0, 1.0));
    return a + (b - a) * fr;
}

void LiveDspEngine::analyseFrame() {
    const long long start = framesDone_ * hop_;
    for (int i = 0; i < span_; ++i) frameBuf_[size_t(i)] = in_[size_t(start + i) & mask_];
    float conf = 0.0f;
    const float f0 = tracker_->estimateFrame(frameBuf_.data(), conf);
    const float dst = corrector_.process(f0);
    srcF0_[size_t(framesDone_) & frameMask_] = f0;
    dstF0_[size_t(framesDone_) & frameMask_] = dst;
    lastSrcF0_ = f0;
    lastDstF0_ = dst;
    ++framesDone_;
}

void LiveDspEngine::placeMarks() {
    const double unvoiced = 0.0058 * sr_;
    for (;;) {
        const float f = f0At(srcF0_, nextMark_);
        const double P = f > 0.0f ? sr_ / f : unvoiced;
        if (nextMark_ + 1.25 * P + 1.0 > double(written_ - 1)) return;
        double m = nextMark_;
        if (f > 0.0f && markCount_ > 0) {
            const long long lo = (long long)(m - 0.25 * P), hi = (long long)(m + 0.25 * P);
            long long best = (long long)m;
            for (long long k = std::max(lo, written_ - (long long)mask_); k <= hi; ++k)
                if (in_[size_t(k) & mask_] > in_[size_t(best) & mask_]) best = k;
            m = double(best);
            const double prev = markPos_[size_t(markCount_ - 1) & markMask_];
            if (m <= prev) m = prev + 0.5 * P;
        }
        markPos_[size_t(markCount_) & markMask_] = m;
        markPer_[size_t(markCount_) & markMask_] = P;
        ++markCount_;
        nextMark_ = m + P;
    }
}

void LiveDspEngine::synthesise() {
    if (markCount_ == 0) return;
    if (nextTs_ < 0.0) nextTs_ = markPos_[0];
    nearest_ = std::max(nearest_, markCount_ - (long long)markMask_);
    for (;;) {
        const double lastMark = markPos_[size_t(markCount_ - 1) & markMask_];
        if (lastMark < nextTs_) return;
        const double ts = nextTs_;
        while (nearest_ + 1 < markCount_ &&
               std::fabs(markPos_[size_t(nearest_ + 1) & markMask_] - ts) <= std::fabs(markPos_[size_t(nearest_) & markMask_] - ts))
            ++nearest_;
        const double Pa = markPer_[size_t(nearest_) & markMask_];
        const float fs = f0At(srcF0_, ts), fd = f0At(dstF0_, ts);
        double Ps = (fs > 0.0f && fd > 0.0f) ? sr_ / fd : Pa;
        Ps = std::max(Ps, 16.0);
        const float gain = float(std::min(2.0, Ps / Pa));
        const long long ma = (long long)std::llround(markPos_[size_t(nearest_) & markMask_]);
        const long long L = (long long)std::ceil(Pa);
        const long long centre = (long long)std::llround(ts);
        for (long long k = -L; k <= L; ++k) {
            const long long src = ma + k;
            if (src < 0 || src >= written_) continue;
            const float w = float(0.5 + 0.5 * std::cos(M_PI * double(k) / double(L + 1)));
            acc_[size_t(centre + k) & mask_] += gain * w * in_[size_t(src) & mask_];
        }
        nextTs_ += Ps;
    }
}

void LiveDspEngine::process(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        in_[size_t(written_) & mask_] = in[i];
        ++written_;
        if (written_ >= span_ && (written_ - span_) % hop_ == 0) analyseFrame();
        if (written_ % kSynthEvery == 0) {
            placeMarks();
            synthesise();
        }
        const long long o = written_ - 1 - latency_;
        if (o >= 0) {
            float& a = acc_[size_t(o) & mask_];
            out[i] = a;
            a = 0.0f;
        } else {
            out[i] = 0.0f;
        }
    }
}

// ------------------------------------------------------------- LiveNeuralEngine

namespace {
constexpr size_t kHistSize = 1 << 16;
constexpr size_t kFeatRing = 256;
constexpr int kMelRight = kWinSize - (kFftSize - kHopSize) / 2; // 1280: samples after t*hop needed for frame t
} // namespace

LiveNeuralEngine::~LiveNeuralEngine() { stop(); }

void LiveNeuralEngine::stop() {
    running_.store(false);
    if (worker_.joinable()) worker_.join();
}

void LiveNeuralEngine::prepare(double hostRate, int maxBlock, std::shared_ptr<Vocoder> vocoder, Config cfg) {
    stop();
    hostRate_ = hostRate;
    cfg_ = cfg;
    vocoder_ = std::move(vocoder);
    inRs_ = std::make_unique<StreamingResampler>(hostRate, double(kModelSampleRate));
    outRs_ = std::make_unique<StreamingResampler>(double(kModelSampleRate), hostRate);
    inFifo_ = std::make_unique<SpscFifo>(1 << 17);
    outFifo_ = std::make_unique<SpscFifo>(1 << 17);
    scratch_.assign(size_t(std::max(4096, maxBlock * 4)), 0.0f);

    PitchTrackerConfig pc; // 44.1k, hop 512
    tracker_ = std::make_unique<PitchTracker>(pc);
    hist_.assign(kHistSize, 0.0f);
    hist44_ = 0;
    melRing_.assign(kFeatRing * kNumMels, std::log(1e-5f));
    f0Ring_.assign(kFeatRing, 0.0f);
    nextFeat_ = nextOut_ = 0;
    prevTail_.clear();
    frameWin_.assign(kWinSize, 0.0f);
    pitchWin_.assign(size_t(tracker_->frameSpan()), 0.0f);
    corrector_.prepare(kFrameRate);
    {
        std::lock_guard<std::mutex> lock(paramMutex_);
        corrector_.setParams(pendingParams_);
    }

    const int K = cfg_.chunkFrames, R = cfg_.lookaheadFrames;
    const double d1 = double((K + R - 1) * kHopSize + kMelRight + 1); // feature look-ahead
    const double budget = double(K * kHopSize);                       // one chunk of compute time
    const double rsDelay44 = 2.0 * inRs_->latencyInputSamples() + 2.0 * outRs_->latencyInputSamples();
    wait_ = int(std::ceil((d1 + budget + rsDelay44) * hostRate / kModelSampleRate)) + maxBlock + 64;
    latency_ = wait_ + int(std::lround(inRs_->latencyInputSamples() +
                                       outRs_->latencyInputSamples() * hostRate / kModelSampleRate));
    hostIn_ = hostOut_ = owed_ = 0;
    underruns_.store(0);

    if (vocoder_) {
        running_.store(true);
        worker_ = std::thread([this] { workerLoop(); });
    }
}

bool LiveNeuralEngine::setParams(const CorrectionParams& p) {
    std::unique_lock<std::mutex> lock(paramMutex_, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    pendingParams_ = p;
    paramsDirty_.store(true);
    return true;
}

void LiveNeuralEngine::process(const float* in, float* out, int n) {
    if (!running_.load()) {
        std::memcpy(out, in, sizeof(float) * size_t(n));
        return;
    }
    const int produced = inRs_->process(in, n, scratch_.data(), int(scratch_.size()));
    inFifo_->push(scratch_.data(), size_t(produced));
    hostIn_ += n;

    int pre = int(std::clamp<long long>(wait_ - hostOut_, 0, n));
    std::fill(out, out + pre, 0.0f);
    int rest = n - pre;
    while (owed_ > 0 && outFifo_->available() > 0) { // drop samples we already replaced with silence
        float dump[256];
        owed_ -= (long long)outFifo_->pop(dump, size_t(std::min<long long>(owed_, 256)));
    }
    const size_t got = owed_ > 0 ? 0 : outFifo_->pop(out + pre, size_t(rest));
    if (int(got) < rest) {
        std::fill(out + pre + got, out + n, 0.0f);
        underruns_.fetch_add(1);
        owed_ += rest - int(got);
    }
    hostOut_ += n;
}

void LiveNeuralEngine::workerLoop() {
    while (running_.load()) {
        if (paramsDirty_.exchange(false)) {
            std::lock_guard<std::mutex> lock(paramMutex_);
            CorrectionParams p = pendingParams_;
            p.frameRate = kFrameRate;
            corrector_.setParams(p);
        }
        if (!workerStep()) std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

bool LiveNeuralEngine::workerStep() {
    bool did = false;
    float tmp[1024];
    for (;;) {
        const size_t got = inFifo_->pop(tmp, 1024);
        if (got == 0) break;
        for (size_t i = 0; i < got; ++i) hist_[size_t(hist44_ + (long long)i) & (kHistSize - 1)] = tmp[i];
        hist44_ += (long long)got;
        did = true;
    }
    auto sample = [&](long long i) { return (i < 0 || i >= hist44_) ? 0.0f : hist_[size_t(i) & (kHistSize - 1)]; };

    // Per-frame features (mel + corrected f0) as soon as their window is complete.
    const int span = tracker_->frameSpan();
    while (hist44_ >= nextFeat_ * kHopSize + kMelRight) {
        const long long t = nextFeat_;
        const long long melStart = t * kHopSize - (kFftSize - kHopSize) / 2;
        for (int i = 0; i < kWinSize; ++i) frameWin_[size_t(i)] = sample(melStart + i);
        melEx_.computeFrame(frameWin_.data(), &melRing_[size_t(t % kFeatRing) * kNumMels]);
        const long long pStart = t * kHopSize + kHopSize / 2 - span / 2;
        for (int i = 0; i < span; ++i) pitchWin_[size_t(i)] = sample(pStart + i);
        float conf = 0.0f;
        f0Ring_[size_t(t % kFeatRing)] = corrector_.process(tracker_->estimateFrame(pitchWin_.data(), conf));
        ++nextFeat_;
        did = true;
    }

    const int K = cfg_.chunkFrames, R = cfg_.lookaheadFrames, C = cfg_.contextFrames;
    while (nextFeat_ >= nextOut_ + K + R) {
        const long long start = std::max(0LL, nextOut_ - C), end = nextOut_ + K + R;
        const int frames = int(end - start);
        MelSpectrogram mel;
        mel.numFrames = frames;
        mel.data.resize(size_t(frames) * kNumMels);
        f0Chunk_.resize(size_t(frames));
        for (int i = 0; i < frames; ++i) {
            const size_t slot = size_t((start + i) % (long long)kFeatRing);
            std::memcpy(mel.frame(i), &melRing_[slot * kNumMels], sizeof(float) * kNumMels);
            f0Chunk_[size_t(i)] = f0Ring_[slot];
        }
        warpFormants(mel, formant_.load());

        const size_t off = size_t(nextOut_ - start) * kHopSize, len = size_t(K) * kHopSize;
        std::string err;
        if (!vocoder_->synthesize(mel.data.data(), frames, f0Chunk_.data(), vocOut_, err) ||
            vocOut_.size() < off + len) {
            // Keep the stream aligned: fall back to the dry signal for this chunk.
            vocOut_.assign(off + len, 0.0f);
            for (size_t i = 0; i < len; ++i) vocOut_[off + i] = sample(nextOut_ * kHopSize + (long long)i);
        }
        float* seg = vocOut_.data() + off;
        const size_t xf = std::min(prevTail_.size(), len);
        for (size_t i = 0; i < xf; ++i) {
            const float w = float(i + 1) / float(xf + 1);
            seg[i] = prevTail_[i] * (1.0f - w) + seg[i] * w;
        }
        const size_t tailLen = std::min<size_t>(256, size_t(R) * kHopSize);
        if (vocOut_.size() >= off + len + tailLen) prevTail_.assign(seg + len, seg + len + tailLen);
        else prevTail_.clear();

        rsOut_.resize(len * 4 + 64);
        const int n = outRs_->process(seg, int(len), rsOut_.data(), int(rsOut_.size()));
        outFifo_->push(rsOut_.data(), size_t(n));
        nextOut_ += K;
        did = true;
    }
    return did;
}

} // namespace voxforge
