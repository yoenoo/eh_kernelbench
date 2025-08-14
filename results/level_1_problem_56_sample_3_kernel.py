import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_kernel(const scalar_t* __restrict__ input,
                             const scalar_t* __restrict__ weight,
                             scalar_t* __restrict__ output,
                             const int batch_size, const int in_channels,
                             const int height, const int width,
                             const int kernel_h, const int kernel_w,
                             const int out_channels, const int stride_h,
                             const int stride_w, const int pad_h,
                             const int pad_w, const int dilation_h,
                             const int dilation_w) {

    const int H_out = ((height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    const int W_out = ((width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;

    const int output_height = H_out;
    const int output_width = W_out;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int batch = blockIdx.z % batch_size;
    const int out_channel = blockIdx.z / batch_size;

    if (out_channel >= out_channels || row >= output_height || col >= output_width) {
        return;
    }

    scalar_t sum = 0;
    for (int k_h = 0; k_h < kernel_h; ++k_h) {
        for (int k_w = 0; k_w < kernel_w; ++k_w) {
            int h_in = -pad_h + row * stride_h + dilation_h * k_h;
            int w_in = -pad_w + col * stride_w + dilation_w * k_w;
            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                    sum += weight[out_channel * in_channels * kernel_h * kernel_w +
                                 in_ch * kernel_h * kernel_w +
                                 k_h * kernel_w + k_w] *
                           input[batch * in_channels * height * width +
                                 in_ch * height * width +
                                 h_in * width + w_in];
                }
            }
        }
    }
    output[batch * out_channels * H_out * W_out +
           out_channel * H_out * W_out +
           row * W_out + col] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                         int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dilation_h, int dilation_w) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);

    int H_out = ((height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    int W_out = ((width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;

    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());

    dim3 threads(256);
    dim3 blocks(output_width, H_out, batch_size * out_channels);

    conv2d_kernel<float><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, height, width,
        kernel_h, kernel_w, out_channels,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w);

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w);"

conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), 
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights like PyTorch Conv2d
        kh, kw = kernel_size
        weight_shape = (out_channels, in_channels // groups, kh, kw)
        self.weight = nn.Parameter(torch.randn(weight_shape) * 0.01)

        # Ensure the CUDA operator is compiled during initialization
        self.conv2d_op = conv2d_op

    def forward(self, x):
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation

        # Convert weight to contiguous format for CUDA kernel
        weight_cont = self.weight.contiguous()

        return self.conv2d_op.conv2d_cuda(
            x.contiguous(), weight_cont,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w
        )