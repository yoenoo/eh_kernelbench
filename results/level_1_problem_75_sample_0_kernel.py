import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void ConvTranspose2DKernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     const int batch_size,
                                     const int in_channels,
                                     const int out_channels,
                                     const int kernel_h,
                                     const int kernel_w,
                                     const int out_h,
                                     const int out_w,
                                     const int stride_h,
                                     const int stride_w,
                                     const int padding_h,
                                     const int padding_w,
                                     const int dilation_h,
                                     const int dilation_w,
                                     const int groups) {

    const int output_volume = batch_size * out_channels * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_volume) return;

    int w = idx % out_w;
    int h = (idx / out_w) % out_h;
    int c_out = (idx / out_w / out_h) % out_channels;
    int n = idx / out_channels / out_h / out_w;

    const int group = c_out / (out_channels / groups);
    const int c_in_group = (group * in_channels) / groups;
    const int c_out_group = c_out % (out_channels / groups);

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int dilated_kh = kh * dilation_h;
            const int dilated_kw = kw * dilation_w;

            const int h_in = h - padding_h - stride_h * kh;
            const int w_in = w - padding_w - stride_w * kw;

            if (h_in >= 0 && h_in < out_h && w_in >= 0 && w_in < out_w) {
                for (int c_in = 0; c_in < in_channels/groups; ++c_in) {
                    const int input_offset = ((n * in_channels + c_in + c_in_group) * out_h + h_in) * out_w + w_in;
                    const int weight_offset = ((group * out_channels/group + c_out_group) * kernel_h * kernel_w + kh * kernel_w + kw) * (in_channels/groups) + c_in;
                    val += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    const int output_offset = ((n * out_channels + c_out) * out_h + h) * out_w + w;
    atomicAdd(&output[output_offset], val);
}

torch::Tensor conv_transpose_2d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int stride_h, int stride_w,
                                    int padding_h, int padding_w,
                                    int dilation_h, int dilation_w,
                                    int groups) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0) * groups;
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);

    // Compute output dimensions
    const int out_h = (input_h - 1) * stride_h - 2 * padding_h + 
                    dilation_h * (kernel_h - 1) + 1;
    const int out_w = (input_w - 1) * stride_w - 2 * padding_w + 
                    dilation_w * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    const int threads = 256;
    int elements = batch_size * out_channels * out_h * out_w;
    int blocks = (elements + threads - 1) / threads;

    const auto in_channels_f = input.scalar_type();
    AT_DISPATCH_FLOATING_TYPES(in_channels.scalar_type(), "conv_transpose_2d_cuda", ([&] {
        ConvTranspose2DKernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w, out_h, out_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups);
    }));

    return output;
}
"""

conv_transpose_2d_cpp_source = """
torch::Tensor conv_transpose_2d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int stride_h, int stride_w,
                                    int padding_h, int padding_w,
                                    int dilation_h, int dilation_w,
                                    int groups);
"""

conv_transpose_2d = load_inline(
    name="conv_transpose_2d",
    cpp_sources=conv_transpose_2d_cpp_source,
    cuda_sources=conv_transpose_2d_source,
    functions=["conv_transpose_2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), 
                 padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights like PyTorch's ConvTranspose2d
        kh, kw = kernel_size
        self.weight = nn.Parameter(torch.randn(
            in_channels, 
            out_channels // groups, 
            kh, kw))
        # Transpose for deconvolution weights
        self.weight.data = self.weight.data.transpose(0,1).contiguous()

    def forward(self, x):
        return conv_transpose_2d.conv_transpose_2d_cuda(
            x, 
            self.weight, 
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )