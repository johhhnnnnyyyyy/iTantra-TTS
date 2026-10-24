#include <cstddef>
#include <cstdint>
#include <limits>

// Reference only: packed two's-complement nibbles, low nibble first.
// Input is [input_channels, length], weights [output_channels, input_channels, kernel].
// The C++ scalar integers are machine-width; this is not a hardware INT4 kernel.
static int unpack(const std::uint8_t* data, std::size_t index) {
    const unsigned nibble = (data[index / 2] >> (4 * (index % 2))) & 15U;
    return nibble < 8 ? static_cast<int>(nibble) : static_cast<int>(nibble) - 16;
}

// MatMulInteger reference: unsigned activation with scalar zero point, packed
// signed symmetric INT4 weights, row-major [inner, columns], INT32 output.
// INT64 temporaries prevent signed overflow during the reference calculation.
extern "C" int itantra_w4u8_matmul(const std::uint8_t* input,
    const std::uint8_t* weights, std::size_t rows, std::size_t inner,
    std::size_t columns, std::uint8_t input_zero_point, std::int32_t* output) {
    constexpr std::size_t limit = 1000000;
    if (!input || !weights || !output || !rows || !inner || !columns ||
        rows > limit || inner > limit || columns > limit || rows > limit / inner ||
        inner > limit / columns || rows > limit / columns) return -1;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < columns; ++col) {
            std::int64_t sum = 0;
            for (std::size_t k = 0; k < inner; ++k) {
                const int activation = static_cast<int>(input[row * inner + k]) -
                                       static_cast<int>(input_zero_point);
                sum += activation * unpack(weights, k * columns + col);
            }
            if (sum < std::numeric_limits<std::int32_t>::min() ||
                sum > std::numeric_limits<std::int32_t>::max()) return -1;
            output[row * columns + col] = static_cast<std::int32_t>(sum);
        }
    }
    return 0;
}

// Reference W4A8 Conv1D: activations stay signed INT8; weights stay packed INT4.
// The INT64 scalar sum models a safe INT32 dot-product accumulator, not INT4 hardware.
extern "C" int itantra_w4a8_conv1d(const std::int8_t* input,
    const std::uint8_t* weights, std::size_t length, std::size_t input_channels,
    std::size_t output_channels, std::size_t kernel, std::size_t stride,
    std::size_t padding, std::int32_t* output) {
    constexpr std::size_t limit = 1000000;
    if (!input || !weights || !output || !length || !input_channels || !output_channels ||
        !kernel || !stride || length > limit || input_channels > limit ||
        output_channels > limit || kernel > limit || stride > limit || padding > limit ||
        padding > (limit - length) / 2 || length + 2 * padding < kernel ||
        input_channels > limit / kernel ||
        output_channels > limit / (input_channels * kernel) ||
        input_channels > limit / length) return -1;
    const std::size_t output_length = (length + 2 * padding - kernel) / stride + 1;
    if (output_channels > limit / output_length) return -1;
    for (std::size_t oc = 0; oc < output_channels; ++oc) {
        for (std::size_t position = 0; position < output_length; ++position) {
            std::int64_t sum = 0;
            for (std::size_t ic = 0; ic < input_channels; ++ic) {
                for (std::size_t k = 0; k < kernel; ++k) {
                    const std::size_t padded = position * stride + k;
                    if (padded < padding || padded - padding >= length) continue;
                    sum += static_cast<std::int32_t>(input[ic * length + padded - padding]) *
                           unpack(weights, (oc * input_channels + ic) * kernel + k);
                }
            }
            if (sum < std::numeric_limits<std::int32_t>::min() ||
                sum > std::numeric_limits<std::int32_t>::max()) return -1;
            output[oc * output_length + position] = static_cast<std::int32_t>(sum);
        }
    }
    return 0;
}

// Input [input_channels, length], weights [input_channels, output_channels, kernel].
extern "C" int itantra_w4a4_conv_transpose1d(const std::uint8_t* input,
    const std::uint8_t* weights, std::size_t length, std::size_t input_channels,
    std::size_t output_channels, std::size_t kernel, std::size_t stride,
    std::size_t padding, std::int32_t* output) {
    constexpr std::size_t limit = 1000000;
    if (!input || !weights || !output || !length || !input_channels || !output_channels ||
        !kernel || !stride || length > limit || input_channels > limit ||
        output_channels > limit || kernel > limit || stride > limit || padding > limit ||
        input_channels > limit / kernel ||
        output_channels > limit / (input_channels * kernel) ||
        input_channels > limit / length || (length - 1) > limit / stride) return -1;
    const std::size_t unpadded_length = (length - 1) * stride + kernel;
    if (unpadded_length <= 2 * padding) return -1;
    const std::size_t output_length = unpadded_length - 2 * padding;
    if (output_channels > limit / output_length) return -1;
    for (std::size_t oc = 0; oc < output_channels; ++oc) {
        for (std::size_t pos = 0; pos < output_length; ++pos) {
            std::int64_t sum = 0;
            for (std::size_t ic = 0; ic < input_channels; ++ic) {
                for (std::size_t k = 0; k < kernel; ++k) {
                    const std::size_t padded = pos + padding;
                    if (padded < k || (padded - k) % stride != 0) continue;
                    const std::size_t input_pos = (padded - k) / stride;
                    if (input_pos >= length) continue;
                    sum += unpack(input, ic * length + input_pos) *
                           unpack(weights, (ic * output_channels + oc) * kernel + k);
                }
            }
            if (sum < std::numeric_limits<std::int32_t>::min() ||
                sum > std::numeric_limits<std::int32_t>::max()) return -1;
            output[oc * output_length + pos] = static_cast<std::int32_t>(sum);
        }
    }
    return 0;
}

// Row-major [rows, inner] times [inner, columns], packed across rows.
extern "C" int itantra_w4a4_matmul(const std::uint8_t* left, const std::uint8_t* right,
    std::size_t rows, std::size_t inner, std::size_t columns, std::int32_t* output) {
    constexpr std::size_t limit = 1000000;
    if (!left || !right || !output || !rows || !inner || !columns ||
        rows > limit || inner > limit || columns > limit || rows > limit / inner ||
        inner > limit / columns || rows > limit / columns) return -1;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < columns; ++col) {
            std::int64_t sum = 0;
            for (std::size_t k = 0; k < inner; ++k)
                sum += unpack(left, row * inner + k) * unpack(right, k * columns + col);
            if (sum < std::numeric_limits<std::int32_t>::min() ||
                sum > std::numeric_limits<std::int32_t>::max()) return -1;
            output[row * columns + col] = static_cast<std::int32_t>(sum);
        }
    }
    return 0;
}

extern "C" int itantra_w4a4_conv1d(const std::uint8_t* input,
    const std::uint8_t* weights, std::size_t length, std::size_t input_channels,
    std::size_t output_channels, std::size_t kernel, std::size_t stride,
    std::size_t padding, std::int32_t* output) {
    constexpr std::size_t limit = 1000000;
    if (!input || !weights || !output || !length || !input_channels || !output_channels ||
        !kernel || !stride || length > limit || input_channels > limit ||
        output_channels > limit || kernel > limit || padding > limit ||
        length + 2 * padding < kernel || input_channels > limit / kernel ||
        output_channels > limit / (input_channels * kernel) ||
        input_channels > limit / length) return -1;

    const std::size_t output_length = (length + 2 * padding - kernel) / stride + 1;
    if (output_channels > limit / output_length) return -1;
    for (std::size_t oc = 0; oc < output_channels; ++oc) {
        for (std::size_t position = 0; position < output_length; ++position) {
            std::int64_t sum = 0;
            for (std::size_t ic = 0; ic < input_channels; ++ic) {
                for (std::size_t k = 0; k < kernel; ++k) {
                    const std::size_t padded = position * stride + k;
                    if (padded < padding || padded - padding >= length) continue;
                    sum += unpack(input, ic * length + padded - padding) *
                           unpack(weights, (oc * input_channels + ic) * kernel + k);
                }
            }
            if (sum < std::numeric_limits<std::int32_t>::min() ||
                sum > std::numeric_limits<std::int32_t>::max()) return -1;
            output[oc * output_length + position] = static_cast<std::int32_t>(sum);
        }
    }
    return 0;
}
