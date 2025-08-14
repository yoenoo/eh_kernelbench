import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,5> input,
                                     const torch::PackedTensorAccessor<scalar_t,5> weight,
                                     torch::PackedTensorAccessor<scalar_t,5> output,
                                     const int Cout, const int Cin,
                                     const int Kd, const int Kh, const int Kw,
                                     const int batch_size, const int Din, const int Hin, const int Win,
                                     const int Dout, const int Hout, const int Wout,
                                     const int padding_d, const int padding_h, const int padding_w,
                                     const int stride_d, const int stride_h, const int stride_w,
                                     const int dilation_d, const int dilation_h, const int dilation_w) {

    const int c_out = blockIdx.z;
    const int d_in = threadIdx.z;
    const int h_in = threadIdx.y;
    const int w_in = threadIdx.x;

    CUDA_KERNEL_LOOP(index, batch_size * Dout * Hout * Wout) {
        const int wout = index % Wout;
        const int hout = (index / Wout) % Hout;
        const int dout = (index / (Wout * Hout)) % Dout;
        const int batch = index / (Wout * Hout * Dout);

        scalar_t val = 0;
        for (int kd = 0; kd < Kd; ++kd) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    const int d_offset = kd * dilation_d;
                    const int h_offset = kh * dilation_h;
                    const int w_offset = kw * dilation_w;

                    const int d = dout * stride_d - padding_d + d_offset;
                    const int h = hout * stride_h - padding_h + h_offset;
                    const int w = wout * stride_w - padding_w + w_offset;

                    // Check if within input dimensions
                    if (d >=0 && d < Din && h >=0 && h < Hin && w >=0 && w < Win) {
                        for (int cin = 0; cin < Cin; ++cin) {
                            val += input[batch][cin][d][h][w] * weight[c_out][cin][kd][kh][kw];
                        }
                    }
                }
            }
        }
        output[batch][c_out][dout][hout][wout] = val;
    }
}

torch::Tensor conv3d_forward(const torch::Tensor input, const torch::Tensor weight,
                            int padding_d, int padding_h, int padding_w,
                            int stride_d, int stride_h, int stride_w,
                            int dilation_d, int dilation_h, int dilation_w) {

    const int batch_size = input.size(0);
    const int Cin = input.size(1);
    const int Din = input.size(2);
    const int Hin = input.size(3);
    const int Win = input.size(4);

    const int Kd = weight.size(2);
    const int Kh = weight.size(3);
    const int Kw = weight.size(4);

    const int Dout = (Din + 2 * padding_d - dilation_d * (Kd - 1) - 1) / stride_d + 1;
    const int Hout = (Hin + 2 * padding_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
    const int Wout = (Win + 2 * padding_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, weight.size(0), Dout, Hout, Wout}, output_options);

    dim3 threads(32, 8, 1); // Thread configuration example
    dim3 blocks((Dout * Hout * Wout + threads.x - 1) / threads.x, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            weight.size(0), Cin,
            Kd, Kh, Kw,
            batch_size, Din, Hin, Win,
            Dout, Hout, Wout,
            padding_d, padding_h, padding_w,
            stride_d, stride_h, stride_w,
            dilation_d, dilation_h, dilation_w
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward(const torch::Tensor input, const torch::Tensor weight,
                            int padding_d, int padding_h, int padding_w,
                            int stride_d, int stride_h, int stride_w,
                            int dilation_d, int dilation_h, int dilation_w);
"""

conv3d_module = load_inline(
    name="conv3d_module",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's default initialization
        weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2])
        self.weight = nn.Parameter(weight)
        
        if bias:
            bias = torch.randn(out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        output = conv3d_module.conv3d_forward(
            x,
            self.weight,
            self.padding[0], self.padding[1], self.padding[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.dilation[0], self.dilation[1], self.dilation[2]
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output