import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized ConvTranspose2d with optional ReLU activation
conv_transpose_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Constants for block and thread dimensions
constexpr int kBlockDimX = 32;
constexpr int kBlockDimY = 32;

template <typename scalar_t>
__global__ void conv_transpose_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int out_channels, int in_channels, int kernel_h, int kernel_w,
    int out_h, int out_w, int in_h, int in_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    bool with_relu) {

    const int batch_idx = blockIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_x = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_y >= out_h || out_x >= out_w) return;

    const int output_offset = batch_idx * out_channels * out_h * out_w + 
                             0 * out_h * out_w + // channel 0
                             out_y * out_w + out_x;

    scalar_t sum = 0;
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int dh = kh * dilation_h;
                const int dw = kw * dilation_w;
                const int in_y = out_y + dh - padding_h;
                const int in_x = out_x + dw - padding_w;

                if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w) continue;

                const int input_offset = batch_idx * in_channels * in_h * in_w +
                                        in_ch * in_h * in_w +
                                        in_y * in_w + in_x;

                const int weight_offset = in_ch * out_channels * kernel_h * kernel_w +
                                         0 * kernel_h * kernel_w + // output channel 0
                                         kh * kernel_w + kw;

                sum += weight[weight_offset] * input[input_offset];
            }
        }
    }

    if (with_relu) {
        output[output_offset] = fmaxf(sum, 0.0f);
    } else {
        output[output_offset] = sum;
    }
}

torch::Tensor conv_transpose_relu_cuda(torch::Tensor input, torch::Tensor weight,
                                      int out_channels, int out_h, int out_w,
                                      int stride_h, int stride_w,
                                      int padding_h, int padding_w,
                                      int dilation_h, int dilation_w,
                                      bool with_relu) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int in_h = input.size(2);
    const int in_w = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 block(kBlockDimX, kBlockDimY);
    dim3 grid(batch_size,
             (out_h + block.y - 1) / block.y,
             (out_w + block.x - 1) / block.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_relu_cuda", ([&] {
        conv_transpose_relu_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(), weight.data<scalar_t>(), 
            output.data<scalar_t>(),
            out_channels, in_channels, kernel_h, kernel_w,
            out_h, out_w, in_h, in_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            with_relu);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_relu_cpp_source = (
    "torch::Tensor conv_transpose_relu_cuda(torch::Tensor input, torch::Tensor weight,"
    "int out_channels, int out_h, int out_w,"
    "int stride_h, int stride_w,"
    "int padding_h, int padding_w,"
    "int dilation_h, int dilation_w,"
    "bool with_relu);"
)

# Compile the custom CUDA kernel
conv_transpose_relu = load_inline(
    name="conv_transpose_relu",
    cuda_sources=conv_transpose_relu_source,
    cpp_sources=conv_transpose_relu_cpp_source,
    functions=["conv_transpose_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        self.use_relu = True  # Optional activation fusion

        # Initialize CUDA kernel module
        self.custom_conv = conv_transpose_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output shape
        batch_size, _, in_h, in_w = x.shape
        out_channels, in_channels, kernel_h, kernel_w = self.weight.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        effective_kernel_h = dilation_h * (kernel_h - 1) + 1
        effective_kernel_w = dilation_w * (kernel_w - 1) + 1

        out_h = (in_h - 1) * stride_h - 2 * padding_h + effective_kernel_h
        out_w = (in_w - 1) * stride_w - 2 * padding_w + effective_kernel_w

        # Execute custom kernel with optional ReLU fusion
        output = self.custom_conv.conv_transpose_relu_cuda(
            x.cuda(), self.weight.cuda(),
            out_channels, out_h, out_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            self.use_relu
        )
        
        # Add bias if required (can be added inside kernel for further optimization)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1).cuda()
            
        return output