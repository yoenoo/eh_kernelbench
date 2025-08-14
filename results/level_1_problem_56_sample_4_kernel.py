import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution
conv2d_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv2d_custom_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    const int n = output.size(0);
    const int out_channels = output.size(1);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    const int in_channels = input.size(1);

    const int h_out_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int w_out_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z;

    if (h_out_id >= out_h || w_out_id >= out_w || batch_id >= n) {
        return;
    }

    scalar_t val = 0.0;
    for (int i = 0; i < in_channels; ++i) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; ++kw < kernel_w; ) { // Use pre-increment to avoid UB
                int h_in = h_out_id * stride_h - padding_h + kh * dilation_h;
                int w_in = w_out_id * stride_w - padding_w + kw * dilation_w;

                if (h_in >= 0 && h_in < input.size(2) && w_in >=0 && w_in < input.size(3)) {
                    for (int oc = 0; oc < out_channels; ++oc) {
                        val += input[batch_id][i][h_in][w_in] * weight[oc][i][kh][kw];
                    }
                }
            }
        }
    }
    output[batch_id][0][h_out_id][w_out_id] = val;
}

torch::Tensor conv2d_custom(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);

    const int out_channels = weight.size(0);
    
    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (out_h + threads.x - 1) / threads.x,
        (out_w + threads.y - 1) / threads.y,
        batch_size
    );

    const int kernel_h_c = kernel_h;
    const int kernel_w_c = kernel_w;
    const int stride_h_c = stride_h;
    const int stride_w_c = stride_w;
    const int padding_h_c = padding_h;
    const int padding_w_c = padding_w;
    const int dilation_h_c = dilation_h;
    const int dilation_w_c = dilation_w;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_custom", ([&] {
        conv2d_custom_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            kernel_h_c, kernel_w_c,
            stride_h_c, stride_w_c,
            padding_h_c, padding_w_c,
            dilation_h_c, dilation_w_c);
    }));

    return output;
}
"""

conv2d_custom_cpp_source = "torch::Tensor conv2d_custom(torch::Tensor input, torch::Tensor weight, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);"

conv2d_custom_module = load_inline(
    name='conv2d_custom',
    cpp_sources=[conv2d_custom_cpp_source],
    cuda_sources=[conv2d_custom_source],
    functions=['conv2d_custom'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
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
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_h, kernel_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Load custom convolution function
        self.custom_conv = conv2d_custom_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.custom_conv.conv2d_custom(
            x, 
            self.weight,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        
        return output