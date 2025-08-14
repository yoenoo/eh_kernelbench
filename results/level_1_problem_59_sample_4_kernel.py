import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                           \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_kernel(scalar_t *input, scalar_t *weight, scalar_t *output,
                            int batch, int in_channels, int out_channels, 
                            int input_depth, int input_height, int input_width,
                            int kernel_size, int stride, int padding, int dilation,
                            int groups, int output_depth, int output_height, int output_width) {
    
    CUDA_KERNEL_LOOP(output_idx, batch * out_channels * output_depth * output_height * output_width) {
        int w = output_idx % output_width;
        int h = (output_idx / output_width) % output_height;
        int d = (output_idx / (output_width * output_height)) % output_depth;
        int c_out = (output_idx / (output_width * output_height * output_depth)) % out_channels;
        int n = output_idx / (out_channels * output_depth * output_height * output_width);
        
        scalar_t val = 0;
        for (int g = 0; g < groups; ++g) {
            int c_in_group = c_out / (out_channels / groups);
            int c_in_start = c_in_group * (in_channels / groups);
            for (int kd = 0; kd < kernel_size; ++kd) {
                int id = d * stride - padding + kd * dilation;
                if (id < 0 || id >= input_depth) continue;
                for (int kh = 0; kh < kernel_size; ++kh) {
                    int ih = h * stride - padding + kh * dilation;
                    if (ih < 0 || ih >= input_height) continue;
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int iw = w * stride - padding + kw * dilation;
                        if (iw < 0 || iw >= input_width) continue;
                        for (int c_in = c_in_start; c_in < c_in_start + (in_channels / groups); ++c_in) {
                            val += input[n * in_channels * input_depth * input_height * input_width +
                                        c_in * input_depth * input_height * input_width +
                                        id * input_height * input_width +
                                        ih * input_width + iw] *
                                   weight[c_out * kernel_size*kernel_size*kernel_size * (in_channels/groups) +
                                          (kd * kernel_size*kernel_size + kh * kernel_size + kw) * (in_channels/groups) + 
                                          c_in - c_in_start];
                        }
                    }
                }
            }
        }
        output[output_idx] = val;
    }
}

std::vector<int64_t> compute_output_size(int input_size, int kernel_size, int stride, int padding, int dilation) {
    int kernel_effective = dilation * (kernel_size - 1) + 1;
    int output_size = (input_size + 2 * padding - kernel_effective) / stride + 1;
    return {output_size};
}

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, 
                            int kernel_size, int stride, int padding, int dilation, int groups) {
    const auto batch = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);
    const auto out_channels = weight.size(0);

    auto output_depth = compute_output_size(input_depth, kernel_size, stride, padding, dilation)[0];
    auto output_height = compute_output_size(input_height, kernel_size, stride, padding, dilation)[0];
    auto output_width = compute_output_size(input_width, 1, 1, 0, 1)[0]; // since kernel_size is (kernel_size x kernel_size x 1)

    auto output = torch::zeros({batch, out_channels, output_depth, output_height, output_width}, input.options());

    dim3 threadsPerBlock(256);
    dim3 numBlocks((output.numel() + threadsPerBlock.x - 1) / threadsPerBlock.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        conv3d_kernel<scalar_t><<<numBlocks, threadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.contiguous().data<scalar_t>(),
            weight.contiguous().data<scalar_t>(),
            output.data<scalar_t>(),
            batch, in_channels, out_channels,
            input_depth, input_height, input_width,
            kernel_size, stride, padding, dilation,
            groups,
            output_depth, output_height, output_width);
    }));

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, "
    "int kernel_size, int stride, int padding, int dilation, int groups);"
)

conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize convolution weights and parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weight tensor similar to PyTorch's Conv3d
        self.weight = nn.Parameter(
            torch.randn(out_channels, 
                       in_channels // groups,
                       kernel_size, kernel_size, 1)  # 1 for depth dimension
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        output = conv3d_cuda.conv3d_forward(
            x.cuda(), 
            self.weight.cuda(),
            self.kernel_size, self.stride, 
            self.padding, self.dilation,
            self.groups
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        
        return output