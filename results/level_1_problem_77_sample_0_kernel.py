import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_SIZE 16

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(scalar_t *input, scalar_t *weight, scalar_t *output,
    int batch_size, int in_channels, int depth, int height, int width,
    int out_channels, int kernel_size, int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation) {
    
    int d = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.w;

    if (h >= out_height || w >= out_width || d >= out_depth) return;

    scalar_t sum = 0;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int dilated_kd = kd * dilation;
                int dilated_kh = kh * dilation;
                int dilated_kw = kw * dilation;

                int input_d = (d - dilated_kd - padding) / stride;
                int input_h = (h - dilated_kh - padding) / stride;
                int input_w = (w - dilated_kw - padding) / stride;

                if (input_d < 0 || input_d >= depth || 
                    input_h < 0 || input_h >= height ||
                    input_w < 0 || input_w >= width) {
                    continue;
                }

                for (int ich = 0; ich < in_channels; ++ich) {
                    sum += input[ich * depth * height * width + input_d * height * width + input_h * width + input_w] *
                        weight[channel * kernel_size * kernel_size * kernel_size * in_channels + 
                            ich * kernel_size * kernel_size * kernel_size +
                            kd * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
    }

    int output_index = channel * out_depth * out_height * out_width +
                        d * out_height * out_width + h * out_width + w;
    output[output_index] = sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Output dimensions calculation
    const int out_depth = (depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const int out_height = (height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const int out_width = (width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    torch::Tensor output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(out_width / threads.x + 1, out_height / threads.y + 1, out_depth);
    blocks.z = out_depth;

    // Launch kernel
    conv_transpose3d_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, depth, height, width,
        out_channels, kernel_size, out_depth, out_height, out_width,
        stride, padding, dilation
    );

    return std::make_tuple(output, input, weight);
}
"""

conv_transpose3d_cpp_source = """
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code for ConvTranspose3d
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-g", "-G", "-O3"],
    extra_cuda_cflags=["--expt-extended-lambda"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        # Initialize weights (simplified for demonstration)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        output, _, _ = conv_transpose3d_op.conv_transpose3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output