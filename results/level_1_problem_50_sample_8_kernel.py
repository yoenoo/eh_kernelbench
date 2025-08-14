import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

convolution_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                            torch::PackedTensorAccessor<scalar_t,4> output,
                            torch::PackedTensorAccessor<scalar_t,4> weight,
                            int batch_size, int input_channels, int input_height, int input_width,
                            int out_channels, int kernel_size, int stride, int padding) {

    const int H_out = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (input_width + 2 * padding - kernel_size) / stride + 1;

    CUDA_KERNEL_LOOP(index, batch_size * out_channels * H_out * W_out) {
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int channel_out = (index / W_out / H_out) % out_channels;
        int n = index / (W_out * H_out * out_channels);

        scalar_t val = 0;
        for(int i = 0; i < input_channels; ++i) {
            for(int kh = 0; kh < kernel_size; ++kh) {
                for(int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = -padding + h_out * stride + kh;
                    int w_in = -padding + w_out * stride + kw;
                    if(h_in >=0 && w_in >=0 && h_in < input_height && w_in < input_width) {
                        val += weight[channel_out][i][kh][kw] * input[n][i][h_in][w_in];
                    }
                }
            }
        }
        output[n][channel_out][h_out][w_out] = val;
    }
}

std::tuple<torch::Tensor, torch::Tensor> custom_conv2d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding) {

    const int batch_size = input.size(0);
    const int input_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // output dimensions
    const int H_out = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (input_width + 2 * padding - kernel_size) / stride + 1;

    torch::Tensor output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());

    dim3 blocks(TORCH_GET_BLOCKS(batch_size * out_channels * H_out * W_out));
    dim3 threads(TORCH_GET_THREADS());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv2d_forward", ([&] {
        custom_conv2d_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::GetCurrentCUDAStream()>>>(
            input.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            batch_size, input_channels, input_height, input_width,
            out_channels, kernel_size, stride, padding);
    }));

    return std::make_tuple(output);
}

static auto custom_conv2d = torch::deploy::RegisterOperators::op("custom_conv2d_forward",
    torch::deploy::Registration(
        "custom_conv2d_forward",
        torch::deploy::Registration::Function<void>(
            [] (const torch::deploy::Device& device, torch::deploy::FunctionFlags /* flags */ ) {
                return c10::Dispatcher::singleton()
                    .findSchema({"custom_conv2d_forward", ""})
                    .schema()
                    .toFunction<torch::Tensor(const torch::Tensor&, const torch::Tensor&, int64_t, int64_t)>();
            }),
        {torch::deploy::TensorType(torch::kFloat32), torch::deploy::TensorType(torch::kFloat32)},
        c10::nullopt));
"""

convolution_cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> custom_conv2d_forward(torch::Tensor input, torch::Tensor weight, int64_t stride, int64_t padding);
"""

convolution_op = load_inline(
    name='custom_conv2d',
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=['custom_conv2d_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Original conv parameters
        self.weight = nn.Parameter(torch.randn(96, 3, 11, 11))  # Manually manage weights
        self.stride = 4
        self.padding = 2
        self.kernel_size = 11

    def forward(self, x):
        # Manually apply padding since custom kernel doesn't handle it implicitly
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        output, _ = convolution_op.custom_conv2d_forward(x, self.weight, self.stride, self.padding)
        return output