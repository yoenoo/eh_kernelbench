import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.nn.modules.utils import _single

# Custom 1D Convolution CUDA implementation
conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv1d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_length,
    int out_channels,
    int kernel_size,
    int padding,
    int stride) {

    CUDA_KERNEL_LOOP(output_idx, batch_size * out_channels) {
        int batch = output_idx / out_channels;
        int out_channel = output_idx % out_channels;

        scalar_t val = 0;
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            const scalar_t* w = weight + out_channel * in_channels * kernel_size + in_channel * kernel_size;
            const scalar_t* i = input + batch * in_channels * in_length + in_channel * in_length;

            for (int t = -padding; t < (in_length - padding); t += stride) {
                if (t >=0 && t < in_length) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int k_offset = k;
                        int t_offset = t + k * stride;
                        if (t_offset >= 0 && t_offset < in_length) {
                            val += w[k] * i[t_offset];
                        }
                    }
                }
            }
        }
        output[output_idx] = val;
    }
}

torch::Tensor conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_length}, input.options());

    dim3 blocks(TorchMeta::ceil_div(batch_size * out_channels, 256));
    dim3 threads(256);

    conv1d_forward_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_length,
        out_channels,
        kernel_size,
        padding,
        stride);

    return output;
}
"""

conv1d_cpp_source = """
#include <torch/extension.h>
torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

conv1d = load_inline(name="conv1d",
                   cpp_sources=conv1d_cpp_source,
                   cuda_sources=conv1d_source,
                   functions=["conv1d_forward"],
                   verbose=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = conv1d.conv1d_forward(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out