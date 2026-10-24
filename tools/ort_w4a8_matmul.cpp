// Experimental ONNX Runtime W4A8 CPU op: AVX2 acceleration and portable fallback.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(ITANTRA_DISABLE_AVX2)
#include <immintrin.h>
#endif

namespace {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(ITANTRA_DISABLE_AVX2)
// Only called for an entirely in-bounds stride-one tile and a proven INT32 sum bound.
__attribute__((target("avx2")))
void conv32_avx2(const std::uint8_t* x, const std::uint8_t* weights, std::int32_t* y,
                std::int64_t input_base, std::int64_t input_length, std::int64_t channels,
                std::int64_t kernel, std::int64_t oc, std::int64_t start,
                std::int64_t dilation, int zero) {
    __m256i sum = _mm256_setzero_si256();
    __m256i sum_high = _mm256_setzero_si256();
    __m256i sum_third = _mm256_setzero_si256();
    __m256i sum_fourth = _mm256_setzero_si256();
    const __m256i zp = _mm256_set1_epi32(zero);
    for (std::int64_t ic = 0; ic < channels; ++ic) {
        for (std::int64_t k = 0; k < kernel; ++k) {
            const auto wi = (oc * channels + ic) * kernel + k;
            const unsigned nibble = (weights[wi / 2] >> (4 * (wi % 2))) & 15U;
            const int value = nibble < 8 ? static_cast<int>(nibble) : static_cast<int>(nibble) - 16;
            const auto* ptr = x + input_base + ic * input_length + start + k * dilation;
            const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
            const __m128i bytes_high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + 16));
            const __m256i activations = _mm256_sub_epi32(_mm256_cvtepu8_epi32(bytes), zp);
            const __m256i activations_high = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(bytes, 8)), zp);
            const __m256i activations_third = _mm256_sub_epi32(_mm256_cvtepu8_epi32(bytes_high), zp);
            const __m256i activations_fourth = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(bytes_high, 8)), zp);
            const __m256i weight = _mm256_set1_epi32(value);
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(activations, weight));
            sum_high = _mm256_add_epi32(sum_high, _mm256_mullo_epi32(activations_high, weight));
            sum_third = _mm256_add_epi32(sum_third, _mm256_mullo_epi32(activations_third, weight));
            sum_fourth = _mm256_add_epi32(sum_fourth, _mm256_mullo_epi32(activations_fourth, weight));
        }
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y), sum);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + 8), sum_high);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + 16), sum_third);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + 24), sum_fourth);
}
#endif

struct Kernel {
    OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
        try {
            Ort::KernelContext context(raw);
            const auto a = context.GetInput(0);
            const auto packed = context.GetInput(1);
            const auto zero = context.GetInput(2);
            const auto shape = context.GetInput(3);
            auto a_shape = a.GetTensorTypeAndShapeInfo().GetShape();
            const auto p_info = packed.GetTensorTypeAndShapeInfo();
            const auto z_info = zero.GetTensorTypeAndShapeInfo();
            const auto s_info = shape.GetTensorTypeAndShapeInfo();
            if (a_shape.size() < 2 || p_info.GetShape().size() != 1 ||
                z_info.GetElementCount() != 1 || s_info.GetElementCount() != 2 ||
                s_info.GetShape().size() != 1)
                throw std::invalid_argument("invalid tensor shapes");
            const auto* dims = shape.GetTensorData<std::int64_t>();
            if (dims[0] <= 0 || dims[1] <= 0 || a_shape.back() != dims[0])
                throw std::invalid_argument("incompatible matrix shapes");
            const auto inner = static_cast<std::size_t>(dims[0]);
            const auto cols = static_cast<std::size_t>(dims[1]);
            constexpr std::size_t max_elements = 100000000;
            if (inner > max_elements / cols || inner * cols > max_elements)
                throw std::invalid_argument("weight shape too large");
            const auto weight_count = inner * cols;
            if (p_info.GetElementCount() != (weight_count + 1) / 2)
                throw std::invalid_argument("incorrect packed weight length");
            const auto count = a.GetTensorTypeAndShapeInfo().GetElementCount();
            const auto rows = count / inner;
            if (rows > max_elements / cols)
                throw std::invalid_argument("output shape too large");
            a_shape.back() = dims[1];
            auto out = context.GetOutput(0, a_shape);
            auto* output = out.GetTensorMutableData<std::int32_t>();
            const auto* activations = a.GetTensorData<std::uint8_t>();
            const auto* weights = packed.GetTensorData<std::uint8_t>();
            const int zp = *zero.GetTensorData<std::uint8_t>();
            struct MatrixWork {
                const std::uint8_t* activations;
                const std::uint8_t* weights;
                std::int32_t* output;
                std::size_t inner, cols;
                int zp;
                std::atomic<bool> overflow{false};
            } work{activations, weights, output, inner, cols, zp};
            auto task = [](void* opaque, std::size_t row) {
                auto& w = *static_cast<MatrixWork*>(opaque);
                for (std::size_t col = 0; col < w.cols; ++col) {
                    std::int64_t sum = 0;
                    for (std::size_t k = 0; k < w.inner; ++k) {
                        const std::size_t index = k * w.cols + col;
                        const unsigned nibble = (w.weights[index / 2] >> (4 * (index % 2))) & 15U;
                        const int weight = nibble < 8 ? static_cast<int>(nibble) : static_cast<int>(nibble) - 16;
                        sum += (static_cast<int>(w.activations[row * w.inner + k]) - w.zp) * weight;
                    }
                    if (sum < std::numeric_limits<std::int32_t>::min() ||
                        sum > std::numeric_limits<std::int32_t>::max()) {
                        w.overflow.store(true, std::memory_order_relaxed);
                        return;
                    }
                    w.output[row * w.cols + col] = static_cast<std::int32_t>(sum);
                }
            };
            Ort::ThrowOnError(Ort::GetApi().KernelContext_ParallelFor(raw, task, rows, 0, &work));
            if (work.overflow.load(std::memory_order_relaxed))
                throw std::invalid_argument("integer accumulator overflow");
            return nullptr;
        } catch (const std::exception& e) {
            return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, e.what());
        } catch (...) {
            return Ort::GetApi().CreateStatus(ORT_FAIL, "unexpected custom operator failure");
        }
    }
};

struct Op : Ort::CustomOpBase<Op, Kernel, true> {
    const char* GetName() const { return "MatMulW4A8"; }
    std::size_t GetInputTypeCount() const { return 4; }
    ONNXTensorElementDataType GetInputType(std::size_t i) const {
        return i == 3 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    std::size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(std::size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo*, void** result) const noexcept {
        try { *result = new Kernel; return nullptr; }
        catch (...) { return api.CreateStatus(ORT_FAIL, "kernel allocation failed"); }
    }
};

// Scalar ConvInteger counterpart: 1-D NCL input and OIK weights, packed W4.
struct ConvKernel {
    OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
        try {
            Ort::KernelContext ctx(raw);
            const auto a = ctx.GetInput(0), w = ctx.GetInput(1), z = ctx.GetInput(2);
            const auto shape = ctx.GetInput(3), geometry = ctx.GetInput(4);
            const auto dims = a.GetTensorTypeAndShapeInfo().GetShape();
            if (dims.size() != 3 || w.GetTensorTypeAndShapeInfo().GetShape().size() != 1 ||
                z.GetTensorTypeAndShapeInfo().GetElementCount() != 1 ||
                shape.GetTensorTypeAndShapeInfo().GetElementCount() != 3 ||
                geometry.GetTensorTypeAndShapeInfo().GetElementCount() != 5)
                throw std::invalid_argument("invalid convolution tensor shapes");
            const auto* wd = shape.GetTensorData<std::int64_t>();
            const auto* g = geometry.GetTensorData<std::int64_t>();
            constexpr std::int64_t limit = 1000000;
            for (auto d : dims) if (d <= 0 || d > limit) throw std::invalid_argument("invalid input shape");
            for (int i = 0; i < 3; ++i) if (wd[i] <= 0 || wd[i] > limit) throw std::invalid_argument("invalid weight shape");
            for (int i = 0; i < 5; ++i) if (g[i] < 0 || g[i] > limit) throw std::invalid_argument("invalid convolution geometry");
            const auto stride = g[0], left = g[1], right = g[2], dilation = g[3], groups = g[4];
            if (!stride || !dilation || !groups || dims[1] != wd[1] * groups || wd[0] % groups)
                throw std::invalid_argument("invalid convolution groups or stride");
            const auto effective = dilation * (wd[2] - 1) + 1;
            const auto padded = dims[2] + left + right;
            if (padded < effective) throw std::invalid_argument("convolution kernel exceeds input");
            const auto length_out = (padded - effective) / stride + 1;
            constexpr std::int64_t max_elements = 100000000;
            if (wd[0] > max_elements / wd[1] || wd[0] * wd[1] > max_elements / wd[2] ||
                dims[0] > max_elements / wd[0] || dims[0] * wd[0] > max_elements / length_out)
                throw std::invalid_argument("convolution shape too large");
            const auto weight_count = wd[0] * wd[1] * wd[2];
            if (w.GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<std::size_t>((weight_count + 1) / 2))
                throw std::invalid_argument("incorrect packed weight length");
            auto output = ctx.GetOutput(0, std::vector<std::int64_t>{dims[0], wd[0], length_out});
            auto* y = output.GetTensorMutableData<std::int32_t>();
            const auto* x = a.GetTensorData<std::uint8_t>();
            const auto* weights = w.GetTensorData<std::uint8_t>();
            const int zero = *z.GetTensorData<std::uint8_t>();
            struct Work {
                const std::uint8_t* x;
                const std::uint8_t* weights;
                std::int32_t* y;
                const std::int64_t* wd;
                std::int64_t channels, input_length, output_length, stride, left, dilation, groups;
                int zero;
                std::atomic<bool> overflow{false};
            } work{x, weights, y, wd, dims[1], dims[2], length_out, stride, left, dilation, groups, zero};
            auto task = [](void* opaque, std::size_t index) {
                auto& w = *static_cast<Work*>(opaque);
                const auto batch = static_cast<std::int64_t>(index) / w.wd[0];
                const auto oc = static_cast<std::int64_t>(index) % w.wd[0];
                const auto group = oc / (w.wd[0] / w.groups);
                for (std::int64_t pos = 0; pos < w.output_length; pos += 8) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(ITANTRA_DISABLE_AVX2)
                    if (__builtin_cpu_supports("avx2") && w.stride == 1 &&
                        w.output_length - pos >= 32 && pos >= w.left &&
                        pos + 31 - w.left + (w.wd[2] - 1) * w.dilation < w.input_length &&
                        w.wd[1] * w.wd[2] <= std::numeric_limits<std::int32_t>::max() / 2040) {
                        const auto base = (batch * w.channels + group * w.wd[1]) * w.input_length;
                        auto* destination = w.y + (batch * w.wd[0] + oc) * w.output_length + pos;
                        conv32_avx2(w.x, w.weights, destination, base, w.input_length, w.wd[1],
                                    w.wd[2], oc, pos - w.left, w.dilation, w.zero);
                        pos += 24;
                        continue;
                    }
#endif
                    std::int64_t sums[8] = {};
                    const auto count = std::min<std::int64_t>(8, w.output_length - pos);
                    for (std::int64_t ic = 0; ic < w.wd[1]; ++ic) {
                        const auto input_base = (batch * w.channels + group * w.wd[1] + ic) * w.input_length;
                        for (std::int64_t k = 0; k < w.wd[2]; ++k) {
                            const auto wi = (oc * w.wd[1] + ic) * w.wd[2] + k;
                            const unsigned nibble = (w.weights[wi / 2] >> (4 * (wi % 2))) & 15U;
                            const int value = nibble < 8 ? static_cast<int>(nibble) : static_cast<int>(nibble) - 16;
                            auto pos_in = pos * w.stride - w.left + k * w.dilation;
                            for (std::int64_t j = 0; j < count; ++j, pos_in += w.stride) {
                                if (pos_in >= 0 && pos_in < w.input_length)
                                    sums[j] += (static_cast<int>(w.x[input_base + pos_in]) - w.zero) * value;
                            }
                        }
                    }
                    for (std::int64_t j = 0; j < count; ++j) {
                        if (sums[j] < std::numeric_limits<std::int32_t>::min() || sums[j] > std::numeric_limits<std::int32_t>::max()) {
                            w.overflow.store(true, std::memory_order_relaxed);
                            return;
                        }
                        w.y[(batch * w.wd[0] + oc) * w.output_length + pos + j] = static_cast<std::int32_t>(sums[j]);
                    }
                }
            };
            Ort::ThrowOnError(Ort::GetApi().KernelContext_ParallelFor(
                raw, task, static_cast<std::size_t>(dims[0] * wd[0]), 0, &work));
            if (work.overflow.load(std::memory_order_relaxed))
                throw std::invalid_argument("convolution accumulator overflow");
            return nullptr;
        } catch (const std::exception& e) {
            return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, e.what());
        } catch (...) {
            return Ort::GetApi().CreateStatus(ORT_FAIL, "unexpected convolution failure");
        }
    }
};

struct ConvOp : Ort::CustomOpBase<ConvOp, ConvKernel, true> {
    const char* GetName() const { return "ConvW4A8"; }
    std::size_t GetInputTypeCount() const { return 5; }
    ONNXTensorElementDataType GetInputType(std::size_t i) const {
        return i >= 3 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    std::size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(std::size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo*, void** result) const noexcept {
        try { *result = new ConvKernel; return nullptr; }
        catch (...) { return api.CreateStatus(ORT_FAIL, "kernel allocation failed"); }
    }
};
}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                    const OrtApiBase* base) noexcept {
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        const OrtApi* fallback = base->GetApi(1);
        return fallback ? fallback->CreateStatus(ORT_FAIL, "incompatible ONNX Runtime API version") : nullptr;
    }
    Ort::InitApi(api);
    try {
        static Op op;
        static ConvOp conv_op;
        static Ort::CustomOpDomain domain("org.itantra.experimental");
        static const bool added = (domain.Add(&op), domain.Add(&conv_op), true);
        (void)added;
        return api->AddCustomOpDomain(options, domain);
    } catch (const std::exception& e) {
        return api->CreateStatus(ORT_FAIL, e.what());
    } catch (...) {
        return api->CreateStatus(ORT_FAIL, "custom operator registration failed");
    }
}
