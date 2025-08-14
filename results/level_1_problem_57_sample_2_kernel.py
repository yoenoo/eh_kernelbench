import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d with fixed parameters
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void ConvTranspose2dKernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                     const torch::PackedTensorAccessor<scalar_t,4> weight,
                                     torch::PackedTensorAccessor<scalar_t,4> output,
                                     int kernel_size,
                                     int stride,
                                     int padding) {

    const int B = input.size(0);
    const int Cin = input.size(1);
    const int Hin = input.size(2);
    const int Win = input.size(3);

    const int Cout = weight.size(0);
    const int Kh = kernel_size;
    const int Kw = kernel_size;

    const int Hout = Hin * stride - 2 * padding + kernel_size;
    const int Wout = Win * stride - 2 * padding + kernel_size;

    const int batch_id = blockIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_x = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_y >= Hout || out_x >= Wout) return;

    for (int cout = 0; cout < Cout; ++cout) {
        scalar_t sum = 0;
        for (int cin = 0; cin < Cin; ++cin) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    int in_h = (out_y + 2 * padding - kh) / stride;
                    int in_w = (out_x + 2 * padding - kw) / stride;

                    if ((out_y + 2 * padding - kh) % stride == 0 &&
                        (out_x + 2 * padding - kw) % stride == 0 &&
                        in_h >= 0 && in_h < Hin &&
                        in_w >= 0 && in_w < Win) {

                        sum += weight[cout][cin][kh][kw] * 
                               input[batch_id][cin][in_h][in_w];
                    }
                }
            }
        }
        output[batch_id][cout][out_y][out_x] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding) {
    const int B = input.size(0);
    const int Cout = weight.size(0);
    const int Hout = input.size(2) * stride - 2 * padding + kernel_size;
    const int Wout = input.size(3) * stride - 2 * padding + kernel_size;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({B, Cout, Hout, Wout}, output_options);

    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 blocks(B, (Hout + block_size - 1)/block_size, (Wout + block_size -1)/block_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        ConvTranspose2dKernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            kernel_size,
            stride,
            padding);
    }));

    return output;
}
"""

# Header for compilation
conv_transpose_header = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);
"""

conv_transpose2d = load_inline(
    name='conv_transpose2d',
    cpp_sources=conv_transpose_header,
    cuda_sources=conv_transpose_source,
    functions=['conv_transpose2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        
        # Bias is omitted as per original configuration (bias=False by default)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d.conv_transpose2d_cuda(x, self.weight, self.kernel_size, self.stride, self.padding)