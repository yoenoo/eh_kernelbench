import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void custom_conv3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,5,torch::DefaultPtrTraits> input,
                                            const torch::PackedTensorAccessor<scalar_t,5,torch::DefaultPtrTraits> weight,
                                            torch::PackedTensorAccessor<scalar_t,5,torch::DefaultPtrTraits> output,
                                            int batch_size, int in_channels, int out_channels,
                                            int input_depth, int input_height, int input_width,
                                            int kernel_depth, int kernel_height, int kernel_width,
                                            int stride, int padding_depth, int padding_height, int padding_width,
                                            int dilation_depth, int dilation_height, int dilation_width) {
    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * output_depth * output_height * output_width) {
        int depth = output_index % output_depth;
        int height = (output_index / output_depth) % output_height;
        int width = (output_index / (output_depth * output_height)) % output_width;
        int batch = output_index / (output_depth * output_height * output_width);

        scalar_t sum = 0;
        for (int f = 0; f < in_channels; f++) {
            for (int kd = 0; kd < kernel_depth; kd++) {
                int id_depth = depth * stride + padding_depth - kd * dilation_depth;
                if (id_depth < 0 || id_depth >= input_depth) continue;
                for (int kh = 0; kh < kernel_height; kh++) {
                    int id_height = height * stride + padding_height - kh * dilation_height;
                    if (id_height < 0 || id_height >= input_height) continue;
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int id_width = width * stride + padding_width - kw * dilation_width;
                        if (id_width < 0 || id_width >= input_width) continue;
                        sum += input[batch][f][id_depth][id_height][id_width] *
                               weight[f][kd][kh][kw];
                    }
                }
            }
        }
        output[batch][output_index % out_channels][depth][height][width] = sum;
    }
}

torch::Tensor custom_conv3d_forward(torch::Tensor input, torch::Tensor weight,
                                   int stride, std::array<int,3> padding, std::array<int,3> dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0); // Assuming weight is [out_channels, in_channels, ...]
    const int kernel_depth = weight.size(2);
    const int kernel_height = weight.size(3);
    const int kernel_width = weight.size(4);

    // Compute output dimensions
    const int output_depth = (input.size(2) + 2 * padding[0] - dilation[0] * (kernel_depth - 1) - 1) / stride + 1;
    const int output_height = (input.size(3) + 2 * padding[1] - dilation[1] * (kernel_height - 1) - 1) / stride + 1;
    const int output_width = (input.size(4) + 2 * padding[2] - dilation[2] * (kernel_width - 1) - 1) / stride + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, output_options);

    int blocks = (batch_size * out_channels * output_depth * output_height * output_width + 512 - 1) / 512;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv3d_forward", ([&] {
        custom_conv3d_forward_kernel<scalar_t><<<blocks, 512>>>(
            input.packed_accessor<scalar_t,5,torch::DefaultPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::DefaultPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::DefaultPtrTraits>(),
            batch_size, in_channels, out_channels,
            input.size(2), input.size(3), input.size(4),
            kernel_depth, kernel_height, kernel_width,
            stride, padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2]);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor custom_conv3d_forward(torch::Tensor input, torch::Tensor weight,
                                   int stride, std::array<int,3> padding, std::array<int,3> dilation);
"""

conv3d_op = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["custom_conv3d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We will use PyTorch's parameter storage but replace the convolution operation
        self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation, dilation)
        else:
            self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Match PyTorch's default init
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv3d_op.custom_conv3d_forward(
            x, self.weight, 
            self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output