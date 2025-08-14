import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused convolution and ReLU CUDA kernel
conv_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/allocator.h>
#include <ATen/native/convolution_utils.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                                  \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);     \
       i += blockDim.x * gridDim.x)

using at::Half;
using at::Tensor;

using at::cuda::detail::TensorShape;
using namespace at::native;

template <typename scalar_t>
__global__ void conv2d_relu_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t batches,
    int64_t input_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_channels,
    int64_t kernel_h,
    int66 kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {

    int output_size = batches * output_channels * (output_height) * (output_width);
    CUDA_1D_KERNEL_LOOP(index, output_size) {
        int w_out = index % output_width;
        int h_out = (index / output_width) % output_height;
        int c_out = (index / (output_width * output_height)) % output_channels;
        int n = index / (output_channels * output_height * output_width);

        scalar_t sum = 0;
        for (int k = 0; k < input_channels; ++k) {
            for (int ky = 0; ky < kernel_h; ++ky) {
                for (int kx = 0; kx < kernel_w; ++kx) {
                    int h_in = -padding_h + h_out * stride_h + ky * dilation_h;
                    int w_in = -padding_w + w_out * stride_w + kx * dilation_w;
                    if (h_in >= 0 && h_in < input_height && w_in >=0 && w_in < input_width) {
                        sum += weight[c_out * kernel_h * kernel_w * input_channels + k * kernel_h * kernel_w + ky * kernel_w + kx] *
                            input[n * input_channels * input_height * input_width + k * input_height * input_width + 
                                (h_in) * input_width + (w_in)];
                    }
                }
            }
        }
        output[index] = fmax(sum, (scalar_t)0); // apply ReLU
    }
}

int conv2d_relu_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_channels = weight.size(0);

    int output_h = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_w = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    output.resize_({batch_size, out_channels, output_h, output_w});

    dim3 gridDim(1);
    dim3 blockDim(1024);
    // Launch kernel
    conv2d_relu_forward_kernel<<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w);
    return 1;
}

"""

# Compile the fused kernel
conv_relu = load_inline(
    name="conv_relu",
    cuda_sources=conv_relu_source,
    functions=["conv2d_relu_forward_cuda"],
    verbose=True,
    extra_cflags=["-gencode=arch=compute_80,code=sm_80", "-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size))
        # Note: Bias not implemented in this example for simplicity, can be added later
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x):
        # Prepare output tensor
        # Note: This implementation assumes no padding or dilation for simplicity,
        # Need to generalize based on actual parameters in a production scenario
        # Currently placeholder, actual computation uses the kernel with correct parameters
        output = conv_relu.conv2d_relu_forward_cuda(
            x,
            self.weight,
            torch.empty_like(x),  # This is placeholder, kernel will handle resizing
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation
        )
        return output