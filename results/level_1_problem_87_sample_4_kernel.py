import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 1x1 pointwise convolution using matrix multiplication
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for pointwise convolution (GEMM based)
template <typename scalar_t>
__global__ void pointwise_conv_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int spatial
) {
    int batch = blockIdx.x;
    int output_channel = blockIdx.y;
    int spatial_pos = threadIdx.x;

    scalar_t sum = 0.0;
    for (int i = 0; i < in_channels; ++i) {
        sum += input[batch * in_channels * spatial + i * spatial + spatial_pos] *
               weight[output_channel * in_channels + i];
    }
    output[batch * out_channels * spatial + output_channel * spatial + spatial_pos] = sum;
}

std::tuple<torch::Tensor> pointwise_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    bool has_bias,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height = input.size(2);
    const int width = input.size(3);
    const int spatial = height * width;

    auto output = torch::empty({batch_size, out_channels, height, width}, input.options());

    dim3 threads(std::min(spatial, 1024));
    dim3 blocks(batch_size, out_channels);

    const int block_size = threads.x;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv_cuda", ([&] {
        pointwise_conv_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            spatial
        );
    }));

    if (has_bias) {
        output.add_(bias.view(1, -1, 1, 1));
    }

    cudaDeviceSynchronization();
    return output;
}
"""

# Compile the CUDA kernel
pointwise_conv = load_inline(
    name="pointwise_conv",
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.pointwise_conv = pointwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = (x, self.weight, self.bias is not None, self.bias if self.bias is not None else x.new_zeros(0))
        return self.pointwise_conv.pointwise_conv_cuda(*inputs)[0]