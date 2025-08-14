import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized convolution with optional ReLU
convolution_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_kernel(const scalar_t* __restrict__ input,
                             const scalar_t* __restrict__ weight,
                             scalar_t* output,
                             const int batch_size,
                             const int in_channels,
                             const int out_channels,
                             const int kernel_h,
                             const int kernel_w,
                             const int input_h,
                             const int input_w,
                             const int output_h,
                             const int output_w,
                             const int stride,
                             const int padding,
                             const int dilation,
                             const bool use_relu) {

    const int K = kernel_h * kernel_w;
    const int ic_per_out = in_channels / weight.shape[1]; // groups handling

    // Thread and block indices
    const int batch_idx = blockIdx.x;
    const int out_y = blockIdx.y;
    const int out_x = blockIdx.z * blockDim.y + threadIdx.y;
    const int out_channel = threadIdx.x;

    if (out_x >= output_w || out_channel >= out_channels) return;

    // Output feature map coordinates
    scalar_t val = 0;
    const int in_channel_group = out_channel / (out_channels / weight.shape[0]);
    const int in_channel_start = in_channel_group * ic_per_out;

    for (int kernel_point = 0; kernel_point < K; kernel_point++) {
        const int ky = kernel_point / kernel_w;
        const int kx = kernel_point % kernel_w;

        // Compute input coordinates
        const int in_y = out_y * stride + ky * dilation - padding;
        const int in_x = out_x * stride + kx * dilation - padding;

        // Boundary check
        if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
            for (int in_c = in_channel_start; in_c < in_channel_start + ic_per_out; in_c++) {
                scalar_t w = weight[out_channel * ic_per_out + in_c - in_channel_start][kernel_point];
                scalar_t i = input[batch_idx * in_channels * input_h * input_w +
                                  in_c * input_h * input_w +
                                  in_y * input_w + in_x];
                val += w * i;
            }
        }
    }

    // Apply activation if required
    if (use_relu) val = val > 0 ? val : 0;

    // Write output
    output[batch_idx * out_channels * output_h * output_w +
           out_channel * output_h * output_w +
           out_y * output_w + out_x] = val;
}

std::tuple<torch::Tensor> conv2d_cuda(torch::Tensor input,
                                     torch::Tensor weight,
                                     int stride,
                                     int padding,
                                     int dilation,
                                     bool use_relu) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int input_h = input.size(2);
    const int input_w = input.size(3);

    const int output_h = (input_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    const dim3 threads(32, 8); // Tuned thread configuration
    const dim3 blocks(batch_size, output_h, (output_w + threads.y - 1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            input_h,
            input_w,
            output_h,
            output_w,
            stride,
            padding,
            dilation,
            use_relu);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input,
                         torch::Tensor weight,
                         int stride,
                         int padding,
                         int dilation,
                         bool use_relu);
"""

conv2d_op = load_inline(name='custom_conv2d',
                       cpp_sources=cpp_source,
                       cuda_sources=convolution_kernel,
                       functions=['conv2d_cuda'],
                       verbose=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Only weights used for convolution (bias handled in separate layer if needed)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        # Get weight tensor in correct format
        weight = self.weight
        # Run custom convolution (no activation by default)
        output = conv2d_op.conv2d_cuda(x, weight, self.stride, self.padding, self.dilation, use_relu=False)
        if self.bias:
            # Handle bias here (could be fused into kernel for better performance)
            output = output + self.bias.view(1, -1, 1, 1)
        return output