#include "voxforge/Vocoder.h"

#ifdef VOXFORGE_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cstring>
#include <thread>
#endif

namespace voxforge {

#ifdef VOXFORGE_WITH_ONNX

namespace {

class NsfHifiGanOnnx final : public Vocoder {
public:
    NsfHifiGanOnnx(const std::string& path, int threads)
        : env_(ORT_LOGGING_LEVEL_WARNING, "voxforge") {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads > 0 ? threads : int(std::max(1u, std::thread::hardware_concurrency() / 2)));
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(__APPLE__) && defined(VOXFORGE_WITH_COREML)
        // CoreML EP (Apple Neural Engine / GPU) when ORT was built with it.
        opts.AppendExecutionProvider("CoreML", {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"}});
#endif
#ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
#else
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), opts);
#endif
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto name = session_->GetInputNameAllocated(i, alloc);
            auto shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 3) {
                melName_ = name.get();
                melChannelsFirst_ = shape[1] == kNumMels; // [1,128,T] vs [1,T,128]
            } else if (shape.size() == 2) {
                f0Name_ = name.get();
            }
        }
        outName_ = session_->GetOutputNameAllocated(0, alloc).get();
        if (melName_.empty() || f0Name_.empty())
            throw std::runtime_error("model does not look like NSF-HiFiGAN (expected 3-D mel and 2-D f0 inputs)");
    }

    std::string name() const override { return "NSF-HiFiGAN (ONNX)"; }

    bool synthesize(const float* mel, int T, const float* f0, std::vector<float>& out, std::string& error) override {
        try {
            std::vector<float> melBuf(size_t(T) * kNumMels);
            if (melChannelsFirst_) {
                for (int t = 0; t < T; ++t)
                    for (int m = 0; m < kNumMels; ++m) melBuf[size_t(m) * T + t] = mel[size_t(t) * kNumMels + m];
            } else {
                std::memcpy(melBuf.data(), mel, melBuf.size() * sizeof(float));
            }
            std::vector<float> f0Buf(f0, f0 + T);
            const std::vector<int64_t> melShape = melChannelsFirst_ ? std::vector<int64_t>{1, kNumMels, T}
                                                                    : std::vector<int64_t>{1, T, kNumMels};
            const std::vector<int64_t> f0Shape{1, T};
            auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputs[2] = {
                Ort::Value::CreateTensor<float>(memInfo, melBuf.data(), melBuf.size(), melShape.data(), melShape.size()),
                Ort::Value::CreateTensor<float>(memInfo, f0Buf.data(), f0Buf.size(), f0Shape.data(), f0Shape.size())};
            const char* inNames[2] = {melName_.c_str(), f0Name_.c_str()};
            const char* outNames[1] = {outName_.c_str()};
            auto result = session_->Run(Ort::RunOptions{nullptr}, inNames, inputs, 2, outNames, 1);
            const float* wav = result[0].GetTensorData<float>();
            const size_t len = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
            out.assign(wav, wav + len);
            return true;
        } catch (const std::exception& e) {
            error = e.what();
            return false;
        }
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string melName_, f0Name_, outName_;
    bool melChannelsFirst_ = false;
};

} // namespace

std::unique_ptr<Vocoder> loadNsfHifiGan(const std::string& path, std::string& error, int threads) {
    try {
        return std::make_unique<NsfHifiGanOnnx>(path, threads);
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}
bool onnxRuntimeAvailable() { return true; }

#else

std::unique_ptr<Vocoder> loadNsfHifiGan(const std::string&, std::string& error, int) {
    error = "built without ONNX Runtime (configure with -DVOXFORGE_WITH_ONNX=ON)";
    return nullptr;
}
bool onnxRuntimeAvailable() { return false; }

#endif

} // namespace voxforge
