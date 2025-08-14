import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int channels, int input_h, int input_w,
    int kernel_size, int stride, int padding) {
    
    const int B = blockIdx.z;
    const int C = blockIdx.y;
    const int out_h = blockIdx.x;
    const int out_w = threadIdx.x;

    scalar_t sum = 0;
    for (int ki = 0; ki < kernel_size; ++ki) {
        for (int kj = 0; kj < kernel_size; ++kj) {
            int h_in = out_h * stride + ki - padding;
            int w_in = out_w * stride + kj - padding;
            if (h_in >=0 && h_in < input_h && w_in >=0 && w_in < input_w) {
                sum += input[B][C][h_in][w_in] * weight[C][0][ki][kj];
            }
        }
    }
    output[B][C][out_h][out_w] = sum;
}

std::vector<int64_t> output_size(int64_t batch_size, int64_t channels, int64_t input_h, int64_t input_w, 
                                int64_t kernel_size, int64_t stride, int64_t padding) {
    int64_t output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int64_t output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    return {batch_size, channels, output_h, output_w};
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                  int kernel_size, int stride, int padding) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);

    auto output_dims = output_size(batch_size, channels, input_h, input_w, kernel_size, stride, padding);
    auto output = torch::empty(output_dims, input.options());

    dim3 threads(32); // Adjust thread block size based on output width
    dim3 blocks(input_dims[2], channels, batch_size); // blockIdx.x=out_h, blockIdx.y=channel, blockIdx.z=batch

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, channels, input_h, input_w,
            kernel_size, stride, padding
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
#include <torch/extension.h>

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_size, int stride, int padding);
"""

# Load the CUDA kernel
depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=[depthwise_conv_cpp_source],
    cuda_sources=[depthwise_conv_source],
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = depthwise_conv.depthwise_conv2d_cuda(x, self.weight, self.kernel_size, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output