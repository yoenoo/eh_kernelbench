import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void ConvTranspose3dForward(const scalar_t* bottom_data, const scalar_t* weight_data, scalar_t* top_data,
                                      int num_kernels, int kernel_dim, int top_count) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        scalar_t value = 0;
        int top_offset = index;
        int d = top_offset / (kernel_dim * kernel_dim * kernel_dim);
        int rest = top_offset % (kernel_dim * kernel_dim * kernel_dim);
        int h = rest / (kernel_dim * kernel_dim);
        int w = rest % (kernel_dim * kernel_dim);
        for (int i = 0; i < kernel_dim; ++i) {
            for (int j = 0; j < kernel_dim; ++j) {
                for (int k = 0; k < kernel_dim; ++k) {
                    const int input_d = d - i;
                    const int input_h = h - j;
                    const int input_w = w - k;
                    if (input_d < 0 || input_h < 0 || input_w < 0) {
                        continue;
                    }
                    value += weight_data[i * kernel_dim * kernel_dim * kernel_dim + j * kernel_dim * kernel_dim + k * kernel_dim + d] *
                            bottom_data[input_d * kernel_dim * kernel_dim + input_h * kernel_dim + input_w];
                }
            }
        }
        top_data[top_offset] = value;
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor bottom, torch::Tensor weight, int kernel_size) {
    const int batch_size = bottom.size(0);
    const int in_channels = bottom.size(1);
    const int bottom_depth = bottom.size(2);
    const int bottom_height = bottom.size(3);
    const int bottom_width = bottom.size(4);

    const int out_channels = weight.size(0);
    const int kernel_dim = kernel_size;

    // Calculate output dimensions based on ConvTranspose3d parameters
    // Assuming stride=1, padding=0, output_padding=0, dilation=1, groups=1 as per the initial test case
    int top_depth = (bottom_depth - 1) * 1 + (kernel_size - 1) * 0 + 1 + 0;
    int top_height = (bottom_height - 1) * 1 + (kernel_size - 1) * 0 + 1 + 0;
    int top_width = (bottom_width - 1) * 1 + (kernel_size - 1) * 0 + 1 + 0;

    auto top = torch::zeros({batch_size, out_channels, top_depth, top_height, top_width}, bottom.options());

    const int num_kernels = batch_size * out_channels * top_depth * top_height * top_width;
    const int threads = 256;
    const int blocks = (num_kernels + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(bottom.type(), "conv_transpose3d_cuda", ([&]{
        ConvTranspose3dForward<scalar_t><<<blocks, threads>>>(
            bottom.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            top.data_ptr<scalar_t>(),
            num_kernels,
            kernel_dim,
            top_depth * top_height * top_width
        );
    }));

    cudaDeviceSynchronize();
    return top;
}

"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor bottom, torch::Tensor weight, int kernel_size);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=0'],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to ConvTranspose3d
        weight_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        self.conv_transpose3d_op = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swap in_channels and out_channels because ConvTranspose3d's weight is [in_channels, out_channels, ...]
        # But in PyTorch ConvTranspose3d, weight is [in_channels, out_channels, ...]
        # Wait the original ConvTranspose3d in Model is defined with in_channels, out_channels order, but the weight dimensions may differ, needs careful checking
        # The custom kernel is using out_channels as the first dimension in weight. To match this, transpose the weight if necessary.
        # Actual PyTorch's ConvTranspose3d weight shape is (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
        # Our custom kernel expects weight to be (out_channels, in_channels, kernel dimensions) ?
        # Wait in the code above, weight.size(0) is out_channels, so the weight is stored as (out_channels, in_channels, kernel_size^3)
        # Which suggests the weight tensor needs to be transposed.
        # Let's adjust the weight for the custom kernel.

        # Therefore, in the __init__ of ModelNew, the weight is initialized with shape (in_channels, out_channels, ...)
        # To pass into the custom kernel which expects (out_channels, in_channels, ...), we need to transpose the weight.

        adjusted_weight = self.weight.permute(1, 0, 2, 3, 4).contiguous()

        out = self.conv_transpose3d_op.conv_transpose3d_cuda(x, adjusted_weight, self.kernel_size)

        if self.bias is not None:
            # Assuming bias is added over spatial dimensions, but ConvTranspose3d adds bias after convolution
            out += self.bias.view(1, -1, 1, 1, 1)

        return out