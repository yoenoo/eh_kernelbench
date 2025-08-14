import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D Transposed Convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

at::Tensor conv_transpose_3d_cuda(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
                                  int stride_d, int stride_h, int stride_w,
                                  int padding_d, int padding_h, int padding_w,
                                  int output_padding_d, int output_padding_h, int output_padding_w,
                                  int groups) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int id = input.size(2);
    const int ih = input.size(3);
    const int iw = input.size(4);
    const int out_channels = weight.size(0);
    const int kd = weight.size(2);
    const int kh = weight.size(3);
    const int kw = weight.size(4);

    // Calculate output dimensions
    const int od = (id - 1) * stride_d - 2 * padding_d + kd + output_padding_d;
    const int oh = (ih - 1) * stride_h - 2 * padding_h + kh + output_padding_h;
    const int ow = (iw - 1) * stride_w - 2 * padding_w + kw + output_padding_w;

    auto output = at::empty({batch_size, out_channels, od, oh, ow}, input.options());

    const int num_kernels = batch_size * out_channels;
    dim3 blocks((num_kernels + 255) / 256, 1, 1);
    dim3 threads(256, 1, 1);

    AT_CUDA_KERNELLauncher(conv_transpose_3d_kernel, 
                          input.data_ptr<float>(), 
                          weight.data_ptr<float>(),
                          bias.data_ptr<float>(),
                          output.data_ptr<float>(),
                          batch_size, in_channels, out_channels,
                          id, ih, iw, kd, kh, kw,
                          stride_d, stride_h, stride_w,
                          padding_d, padding_h, padding_w,
                          output_padding_d, output_padding_h, output_padding_w,
                          groups,
                          od, oh, ow, 
                          blocks, threads, 0, at::globalContext().getStream());

    return output;
}

extern "C" 
at::Tensor conv_transpose_3d_forward(const at::Tensor& input,
                                     const at::Tensor& weight,
                                     const at::Tensor& bias,
                                     int stride_d,
                                     int stride_h,
                                     int stride_w,
                                     int padding_d,
                                     int padding_h,
                                     int padding_w,
                                     int output_padding_d,
                                     int output_padding_h,
                                     int output_padding_w,
                                     int groups) {
    return conv_transpose_3d_cuda(input, weight, bias,
                                stride_d, stride_h, stride_w,
                                padding_d, padding_h, padding_w,
                                output_padding_d, output_padding_h, output_padding_w,
                                groups);
}
"""

conv_transpose_3d_header = """
at::Tensor conv_transpose_3d_forward(const at::Tensor& input,
                                     const at::Tensor& weight,
                                     const at::Tensor& bias,
                                     int stride_d,
                                     int stride_h,
                                     int stride_w,
                                     int padding_d,
                                     int padding_h,
                                     int padding_w,
                                     int output_padding_d,
                                     int output_padding_h,
                                     int output_padding_w,
                                     int groups);
"""

module = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_header,
    cuda_sources=conv_transpose_3d_source,
    functions="conv_transpose_3d_forward",
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride_d, self.stride_h, self.stride_w = stride
        self.padding_d, self.padding_h, self.padding_w = padding
        self.output_padding_d, self.output_padding_h, self.output_padding_w = output_padding
        self.groups = groups
        
        # Initialize weight and bias similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return module.conv_transpose_3d_forward(
                x, 
                self.weight, 
                self.bias, 
                self.stride_d, self.stride_h, self.stride_w,
                self.padding_d, self.padding_h, self.padding_w,
                self.output_padding_d, self.output_padding_h, self.output_padding_w,
                self.groups
            )
        else:
            # Create zero bias tensor if bias is not present
            zero_bias = torch.zeros(self.weight.size(0), device=x.device)
            return module.conv_transpose_3d_forward(
                x, 
                self.weight, 
                zero_bias, 
                self.stride_d, self.stride_h, self.stride_w,
                self.padding_d, self.padding_h, self.padding_w,
                self.output_padding_d, self.output_padding_h, self.output_padding_w,
                self.groups
            )