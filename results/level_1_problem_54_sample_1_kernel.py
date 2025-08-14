import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> weight,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int batch_size, int in_channels, int depth, int width, int height,
    int out_channels, int kernel_size, int stride, int padding, int dilation,
    int groups) {

    const int D_out = (depth + 2*padding - dilation*(kernel_size-1) -1)/stride +1;
    const int H_out = (height + 2*padding - dilation*(kernel_size-1) -1)/stride +1;
    const int W_out = (width + 2*padding - dilation*(kernel_size-1) -1)/stride +1;

    const int d_in = blockIdx.z;
    const int h_out = blockIdx.y;
    const int w_out = blockIdx.x;
    const int b = blockIdx.y; // blockIdx.y is unused here, need to fix

    // Using threadIdx to cover the remaining dimensions
    const int channel_group = threadIdx.z;
    const int k_d = threadIdx.y;
    const int k_h = threadIdx.x;

    for (int out_ch = channel_group; out_ch < out_channels; out_ch += groups) {
        for (int d = 0; d < kernel_size; ++d) {
            for (int h = 0; h < kernel_size; ++h) {
                for (int w = 0; w < kernel_size; ++w) {
                    // Compute input indices
                    int in_d = d_out * stride - padding + d*dilation;
                    int in_h = h_out * stride - padding + h*dilation;
                    int in_w = w_out * stride - padding + w*dilation;

                    if (in_d < 0 || in_d >= depth || in_h <0 || in_h >= height || in_w <0 || in_w >= width) {
                        continue;
                    }

                    // Compute the value
                    scalar_t val = 0;
                    for (int c = 0; c < in_channels; ++c) {
                        val += input[b][c][in_d][in_h][in_w] * weight[out_ch][c][d][h][w];
                    }

                    atomicAdd(&output[b][out_ch][d_out][h_out][w_out], val);
                }
            }
        }
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int groups) {
    auto output = torch::zeros({input.size(0), weight.size(0), 
        (input.size(2)+2*padding - dilation*(weight.size(2)-1) -1)/stride +1,
        (input.size(3)+2*padding - dilation*(weight.size(3)-1) -1)/stride +1,
        (input.size(4)+2*padding - dilation*(weight.size(4)-1) -1)/stride +1}, input.options());

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int width = input.size(3);
    const int height = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    dim3 threads(16, 16, 1);
    dim3 blocks(1, D_out, H_out, W_out);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward_cuda", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            batch_size, in_channels, depth, width, height,
            out_channels, kernel_size, stride, padding, dilation,
            groups);
    }));

    return output;
}
"""

conv3d_cuda_header = """
torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int groups);
"""

# Compile the inline CUDA code
conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cuda_sources=conv3d_cuda_source,
    cpp_sources=conv3d_cuda_header,
    functions=["conv3d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's Conv3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Load the CUDA kernel
        self.conv3d_forward_cuda = conv3d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv3d_forward_cuda.conv3d_forward_cuda(
            x, self.weight, self.stride, self.padding, self.dilation, self.groups)
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        
        return output