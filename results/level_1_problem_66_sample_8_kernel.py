import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom Conv3D CUDA kernel
conv3d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
                                     const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
                                     torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
                                     int batch_size, int in_channels, int out_channels,
                                     int input_depth, int input_height, int input_width,
                                     int kernel_depth, int kernel_height, int kernel_width,
                                     int stride_d, int stride_h, int stride_w,
                                     int padding_d, int padding_h, int padding_w,
                                     int dilation_d, int dilation_h, int dilation_w) {
    const int pl = output_width;
    const int pk = output_height;
    const int pj = output_depth;
    
    const int output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
    const int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    const int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    
    CUDA_KERNEL_LOOP(index, batch_size * out_channels * output_depth * output_height * output_width) {
        int n = index / (out_channels * output_depth * output_height * output_width);
        int c_out = (index / (output_depth * output_height * output_width)) % out_channels;
        int d_out = (index / (output_height * output_width)) % output_depth;
        int h_out = (index / output_width) % output_height;
        int w_out = index % output_width;

        scalar_t val = 0;
        for (int k = 0; k < in_channels; ++k) {
            for (int dj = 0; dj < kernel_depth; ++dj) {
                int di = d_out * stride_d - padding_d + dj * dilation_d;
                if (di < 0 || di >= input_depth) continue;
                
                for (int dk = 0; dk < kernel_height; ++dk) {
                    int hi = h_out * stride_h - padding_h + dk * dilation_h;
                    if (hi < 0 || hi >= input_height) continue;
                    
                    for (int dl = 0; dl < kernel_width; ++dl) {
                        int wi = w_out * stride_w - padding_w + dl * dilation_w;
                        if (wi < 0 || wi >= input_width) continue;
                        
                        val += input[n][k][di][hi][wi] * weight[c_out][k][dj][dk][dl];
                    }
                }
            }
        }
        output[n][c_out][d_out][h_out][w_out] = val;
    }
}

at::Tensor conv3d_forward_cuda(const at::Tensor &input, const at::Tensor &weight,
                              int stride_d, int stride_h, int stride_w,
                              int padding_d, int padding_h, int padding_w,
                              int dilation_d, int dilation_h, int dilation_w) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const auto kernel_depth = weight.size(2);
    const auto kernel_height = weight.size(3);
    const auto kernel_width = weight.size(4);

    const auto output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
    const auto output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    const auto output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    auto output = at::empty({batch_size, weight.size(0), output_depth, output_height, output_width}, input.options());

    const int num_threads = 512;
    int total_elements = batch_size * weight.size(0) * output_depth * output_height * output_width;
    int blocks = (total_elements + num_threads - 1) / num_threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward_cuda", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, num_threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            batch_size, in_channels, weight.size(0),
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w);
    }));

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_forward", &conv3d_forward_cuda, "3D convolution forward");
}
"""

conv3d_cpp_source = """
void conv3d_forward(const torch::Tensor &input, const torch::Tensor &weight,
                   int stride_d, int stride_h, int stride_w,
                   int padding_d, int padding_h, int padding_w,
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1,1,1), padding: tuple = (0,0,0), 
                 dilation: tuple = (1,1,1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride_d, self.stride_h, self.stride_w = stride
        self.padding_d, self.padding_h, self.padding_w = padding
        self.dilation_d, self.dilation_h, self.dilation_w = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv3d_module.conv3d_forward(
            x, self.weight, 
            self.stride_d, self.stride_h, self.stride_w,
            self.padding_d, self.padding_h, self.padding_w,
            self.dilation_d, self.dilation_h, self.dilation_w
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output