import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <stdio.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
                             const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
                             torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
                             int64_t batch_size, int64_t output_channels, int64_t input_channels,
                             int64_t input_depth, int64_t input_height, int64_t input_width,
                             int64_t kernel_depth, int64_t kernel_height, int64_t kernel_width,
                             int64_t stride_d, int64_t stride_h, int64_t stride_w,
                             int64_t padding_d, int64_t padding_h, int64_t padding_w,
                             int64_t dilation_d, int64_t dilation_h, int64_t dilation_w) {

    const int64_t n = batch_size * output_channels;
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    const int64_t w = index % output_channels;
    const int64_t b = index / output_channels;

    int64_t output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
    int64_t output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int64_t output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    for (int64_t d_out = 0; d_out < output_depth; d_out++) {
        for (int64_t h_out = 0; h_out < output_height; h_out++) {
            for (int64_t w_out = 0; w_out < output_width; w_out++) {
                scalar_t val = 0;
                for (int64_t k_d = 0; k_d < kernel_depth; k_d++) {
                    int64_t d = d_out * stride_d - padding_d + k_d * dilation_d;
                    if (d < 0 || d >= input_depth) continue;
                    for (int64_t k_h = 0; k_h < kernel_height; k_h++) {
                        int64_t h = h_out * stride_h - padding_h + k_h * dilation_h;
                        if (h < 0 || h >= input_height) continue;
                        for (int64_t k_w = 0; k_w < kernel_width; k_w++) {
                            int64_t w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                            if (w_in < 0 || w_in >= input_width) continue;
                            for (int64_t c = 0; c < input_channels; c++) {
                                val += input[b][c][d][h][w_in] * 
                                       weight[w][c][k_d][k_h][k_w];
                            }
                        }
                    }
                }
                output[b][w][d_out][h_out][w_out] = val;
            }
        }
    }
}

torch::Tensor conv3d_forward(const torch::Tensor& input, const torch::Tensor& weight, 
                            int64_t stride_d, int64_t stride_h, int64_t stride_w, 
                            int64_t padding_d, int64_t padding_h, int64_t padding_w, 
                            int64_t dilation_d, int64_t dilation_h, int64_t dilation_w) {
    
    const auto batch_size = input.size(0);
    const auto input_channels = input.size(1);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);
    
    const auto output_channels = weight.size(0); // out_channels is the first dim of weight
    const auto kernel_depth = weight.size(2);
    const auto kernel_height = weight.size(3);
    const auto kernel_width = weight.size(4);

    int64_t output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
    int64_t output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int64_t output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    auto output_options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::zeros({batch_size, output_channels, output_depth, output_height, output_width}, output_options);

    auto stream = at::cuda::current_stream();
    const int threads = 1024;
    const int blocks = (batch_size * output_channels + threads - 1) / threads;

    // Use 5D accessor for memory layout
    auto input_acc = input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>();
    auto weight_acc = weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>();
    auto output_acc = output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>();

    conv3d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input_acc, weight_acc, output_acc,
        batch_size, output_channels, input_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w);

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward(const torch::Tensor& input, const torch::Tensor& weight, 
                            int64_t stride_d, int64_t stride_h, int64_t stride_w, 
                            int64_t padding_d, int64_t padding_h, int64_t padding_w, 
                            int64_t dilation_d, int64_t dilation_h, int64_t dilation_w);
"""

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_70,code=sm_70", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Mimic PyTorch Conv3d initialization logic
        kernel_size_ = torch.nn.modules.utils._triple(kernel_size)
        stride_ = torch.nn.modules.utils._triple(stride)
        padding_ = torch.nn.modules.utils._triple(padding)
        dilation_ = torch.nn.modules.utils._triple(dilation)

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels // groups, 
                                               kernel_size_[0], kernel_size_[1], kernel_size_[2])))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Initialize parameters like PyTorch does for fair comparison
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.stride = stride_
        self.padding = padding_
        self.dilation = dilation_
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size_

        # Bind the CUDA function
        self.conv3d_forward = conv3d.conv3d_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get parameters from model
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation

        out = self.conv3d_forward(
            x,
            self.weight,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w
        )

        # Add bias if needed
        if self.bias is not None:
            # Assuming bias addition requires another kernel, but for simplicity add with broadcast
            out += self.bias.view(1, -1, 1, 1, 1)

        return out