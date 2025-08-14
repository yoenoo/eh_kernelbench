import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_custom_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                     const torch::PackedTensorAccessor<scalar_t,4> weight,
                                     torch::PackedTensorAccessor<scalar_t,4> output) {
    const int B = input.size(0);
    const int Cout = weight.size(0);
    const int Cin = weight.size(1);
    const int Kh = weight.size(2);
    const int Kw = weight.size(3);
    const int H = input.size(2);
    const int W = input.size(3);
    
    const int H_out = H - Kh + 1;
    const int W_out = W - Kw + 1;

    const int b = blockIdx.z;
    const int c_out = blockIdx.y;
    const int h_out = blockIdx.x * blockDim.y + threadIdx.y;
    const int w_out = threadIdx.x;

    if (h_out < H_out && w_out < W_out) {
        scalar_t sum = 0;
        for (int c_in = 0; c_in < Cin; ++c_in) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    sum += weight[c_out][c_in][kh][kw] *
                           input[b][c_in][h_out + kh][w_out + kw];
                }
            }
        }
        output[b][c_out][h_out][w_out] = sum;
    }
}

torch::Tensor conv2d_forward(const torch::Tensor input, const torch::Tensor weight) {
    const auto B = input.size(0);
    const auto Cout = weight.size(0);
    const auto H_out = input.size(2) - weight.size(2) + 1;
    const auto W_out = input.size(3) - weight.size(3) + 1;
    
    auto output = torch::zeros({B, Cout, H_out, W_out}, input.options());

    int block_dim_x = 32;
    int block_dim_y = 8;
    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(input.size(3)-weight.size(3)+1, 
              (H_out + block_dim_y - 1)/block_dim_y,
              B * Cout);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&] {
        conv2d_forward_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>()
        );
    }));

    return output;
}
"""

conv2d_custom_header = "torch::Tensor conv2d_forward(const torch::Tensor input, const torch::Tensor weight);"

conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_header,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He initialization
        
    def forward(self, x):
        # The custom kernel currently assumes no padding, stride=1, dilation=1, groups=1, and no bias
        # Hence, applying same assumptions here unless parameters are adapted in the kernel
        return conv2d.conv2d_forward(x, self.weight)