import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int height_in, int width_in,
    int kernel_size, int out_channels_per_group,
    int height_out, int width_out,
    int stride_h, int stride_w, int pad_h, int pad_w) {

    CUDA_KERNEL_LOOP(index, batch_size * in_channels * height_out * width_out) {
        int w_out = index % width_out;
        int h_out = (index / width_out) % height_out;
        int c_in = (index / (height_out * width_out)) % in_channels;
        int n = index / (in_channels * height_out * width_out);

        scalar_t val = 0;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                    val += input[n * in_channels * height_in * width_in + c_in * height_in * width_in + h_in * width_in + w_in] *
                        weight[c_in * kernel_size * kernel_size + (kh * kernel_size + kw)];
                }
            }
        }
        output[index * out_channels_per_group] = val; // Assuming out_channels == in_channels and no bias
    }
}

torch::Tensor depthwise_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size, int stride_h, int stride_w, int pad_h, int pad_w) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height_in = input.size(2);
    const auto width_in = input.size(3);
    const auto out_channels_per_group = weight.size(0)/in_channels; // Assumes weight is [in_channels * kernel_size^2]
    const auto height_out = (height_in + 2 * pad_h - kernel_size) / stride_h + 1;
    const auto width_out = (width_in + 2 * pad_w - kernel_size) / stride_w + 1;

    auto output = torch::zeros({batch_size, in_channels, height_out, width_out}, input.options());

    dim3 blocks(1);
    dim3 threads(512); // Tune this value according to your GPU
    auto total_threads = batch_size * in_channels * height_out * width_out;
    depthwise_conv2d_forward_kernel<float><<<GET_BLOCKS(total_threads), GET_THREADS>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, height_in, width_in,
        kernel_size, out_channels_per_group,
        height_out, width_out,
        stride_h, stride_w, pad_h, pad_w);

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the CUDA kernel
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources="",
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        assert in_channels == out_channels, "Custom depthwise kernel requires out_channels = in_channels"
        assert not bias, "Bias is currently not supported in this custom kernel"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(in_channels, kernel_size * kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Same as PyTorch's default initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_conv2d.depthwise_conv2d_forward_cuda(
            x, self.weight, self.kernel_size, self.stride, self.stride, self.padding, self.padding
        )