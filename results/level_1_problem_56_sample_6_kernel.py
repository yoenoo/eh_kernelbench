import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, float* output,
                             int batch_size, int in_channels, int out_channels,
                             int kernel_h, int kernel_w, int input_h, int input_w,
                             int output_h, int output_w, int stride_h, int stride_w) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if (out_y >= output_h || out_x >= output_w) return;

    for (int out_ch = threadIdx.z; out_ch < out_channels; out_ch += blockDim.z) {
        float sum = 0.0f;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int in_h = out_y * stride_h + k_h;
                int in_w = out_x * stride_w + k_w;
                if (in_h < input_h && in_w < input_w) {
                    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                        sum += input[batch * in_channels * input_h * input_w + in_ch * input_h * input_w +
                                    in_h * input_w + in_w] *
                               weight[out_ch * in_channels * kernel_h * kernel_w + in_ch * kernel_h * kernel_w +
                                      k_h * kernel_w + k_w];
                    }
                }
            }
        }
        output[batch * out_channels * output_h * output_w + out_ch * output_h * output_w +
               out_y * output_w + out_x] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int output_h = (input_h - kernel_h) / stride_h + 1;
    int output_w = (input_w - kernel_w) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(8, 8, 1);
    dim3 blocks(output_w / threads.x + 1, output_h / threads.y + 1, batch_size);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_h, kernel_w, input_h, input_w,
        output_h, output_w, stride_h, stride_w
    );

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w);"
)

# Compile the inline CUDA code for 2D convolution
conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride_h, self.stride_w = stride
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv2d_op.conv2d_cuda(x, self.weight, self.stride_h, self.stride_w)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output