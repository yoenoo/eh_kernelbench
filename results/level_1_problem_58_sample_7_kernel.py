#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output,
                                       int batch_size, int in_channels, int out_channels,
                                       int input_depth, int input_height, int input_width,
                                       int kernel_depth, int kernel_height, int kernel_width,
                                       int stride_d, int stride_h, int stride_w,
                                       int padding_d, int padding_h, int padding_w,
                                       int output_padding_d, int output_padding_h, int output_padding_w) {
    int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int d = (idx / (output_width * output_height)) % output_depth;
    int c_out = (idx / (output_width * output_height * output_depth)) % out_channels;
    int n = idx / (out_channels * output_depth * output_height * output_width);

    float value = 0.0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int input_d = (d - kd - padding_d) / stride_d;
                    int input_h = (h - kh - padding_h) / stride_h;
                    int input_w = (w - kw - padding_w) / stride_w;

                    if (input_d >= 0 && input_d < input_depth &&
                        input_h >= 0 && input_h < input_height &&
                        input_w >= 0 && input_w < input_width &&
                        (d - kd - padding_d) % stride_d == 0 &&
                        (h - kh - padding_h) % stride_h == 0 &&
                        (w - kw - padding_w) % stride_w == 0) {
                        value += weight[((c_out * in_channels + c_in) * kernel_depth + kd) * kernel_height + kh) * kernel_width + kw) *
                                 input[n * in_channels * input_depth * input_height * input_width +
                                       c_in * input_depth * input_height * input_width +
                                       input_d * input_height * input_width +
                                       input_h * input_width + input_w];
                    }
                }
            }
        }
    }
    output[idx] = value;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight,
                                    int stride_d, int stride_h, int stride_w,
                                    int padding_d, int padding_h, int padding_w,
                                    int output_padding_d, int output_padding_h, int output_padding_w) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const auto kernel_depth = weight.size(2);
    const auto kernel_height = weight.size(3);
    const auto kernel_width = weight.size(4);

    const auto out_channels = weight.size(1);

    const int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int threads = 256;
    int blocks = (batch_size * out_channels * output_depth * output_height * output_width + threads - 1) / threads;

    conv_transpose_3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_3d_cuda", &conv_transpose_3d_cuda, "Custom 3D transpose convolution CUDA kernel");
}