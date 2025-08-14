import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int depth_in,
    const int height_in,
    const int width_in,
    const int depth_out,
    const int height_out,
    const int width_out,
    const bool bias_term,
    const scalar_t* __restrict__ bias) {

    const int out_depth = blockIdx.z;
    const int row        = blockIdx.y;
    const int col        = blockIdx.x;
    const int b = threadIdx.z;
    const int out_channel = threadIdx.y;
    const int in_channel_div = threadIdx.x;

    const int total_out_channels = gridDim.y * blockDim.y;
    const int total_out_depth = gridDim.z;
    const int total_threads = blockDim.x * blockDim.y * blockDim.z;

    for (int out_idx = b * total_out_channels * total_out_depth * total_threads
          + out_channel * total_out_depth * total_threads
          + out_depth * total_threads + threadIdx.x + threadIdx.y * blockDim.x
          + threadIdx.z * blockDim.x * blockDim.y;
         out_idx < batch_size * out_channels * depth_out * height_out * width_out;
         out_idx += gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z) {
        
        const int batch = out_idx / (out_channels * depth_out * height_out * width_out);
        int rest = out_idx % (out_channels * depth_out * height_out * width_out);
        const int oc = rest / (depth_out * height_out * width_out);
        rest %= (depth_out * height_out * width_out);
        const int od = rest / (height_out * width_out);
        rest %= (height_out * width_out);
        const int oh = rest / width_out;
        const int ow = rest % width_out;

        scalar_t val = 0;
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int id = od - padding - dilation * kd;
                    if (id < 0 || id >= depth_in) continue;
                    const int ih = oh - padding - dilation * kh;
                    if (ih < 0 || ih >= height_in) continue;
                    const int iw = ow - padding - dilation * kw;
                    if (iw < 0 || iw >= width_in) continue;

                    const int in_offset = batch * in_channels * depth_in * height_in * width_in
                                          + in_channel_div * depth_in * height_in * width_in
                                          + id * height_in * width_in + ih * width_in + iw;
                    
                    const int weight_offset = oc * in_channels * kernel_size * kernel_size * kernel_size
                                              + in_channel_div * kernel_size * kernel_size * kernel_size
                                              + kd * kernel_size * kernel_size + kh * kernel_size + kw;

                    val += input[in_offset] * weight[weight_offset];
                }
            }
        }

        int out_offset = batch * out_channels * depth_out * height_out * width_out
                         + oc * depth_out * height_out * width_out
                         + od * height_out * width_out + oh * width_out + ow;

        if (bias_term) val += bias[oc];
        output[out_offset] = val;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    bool bias_term) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto depth_in = input.size(2);
    const auto height_in = input.size(3);
    const auto width_in = input.size(4);

    const auto out_channels = weight.size(0);
    const auto depth_out = (depth_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const auto height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const auto width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int threads = 256;
    dim3 blocks(
        (width_out + threads - 1) / threads,
        (height_out + threads - 1) / threads,
        (depth_out + threads - 1) / threads
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            depth_in,
            height_in,
            width_in,
            depth_out,
            height_out,
            width_out,
            bias_term,
            bias.data_ptr<scalar_t>());
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, input, weight);
}
"""

conv_transpose3d_cpp_source = """
#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    bool bias_term);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions="conv_transpose3d_forward",
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.bias = bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.constant_(self.bias, 0.0)
        else:
            self.bias = None

    def forward(self, x):
        bias_term = self.bias is not None
        return conv_transpose3d.conv_transpose3d_forward(
            x, self.weight, self.bias if bias_term else x.new_zeros(0), self.stride,
            self.padding, self.dilation, self.kernel_size, bias_term)[0]