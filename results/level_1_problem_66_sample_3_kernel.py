import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int depth, const int height, const int width,
    const int out_channels,
    const int kD, const int kH, const int kW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w) {

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels) {
        const int w = output_idx % width;
        const int h = (output_idx / width) % height;
        const int d = (output_idx / (width * height)) % depth;
        const int channel = (output_idx / (width * height * depth)) % out_channels;
        const int n = output_idx / (out_channels * depth * height * width);

        scalar_t val = 0;
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    const int di = d * stride_d + pad_d - kd * dilation_d;
                    const int hi = h * stride_h + pad_h - kh * dilation_h;
                    const int wi = w * stride_w + pad_w - kw * dilation_w;
                    
                    for (int c = 0; c < in_channels; ++c) {
                        // Bounds checking for input indices
                        if (di < 0 || di >= depth || hi < 0 || hi >= height || wi < 0 || wi >= width) {
                            continue;
                        }
                        val += input[n * in_channels * depth * height * width +
                                  c * depth * height * width +
                                  di * height * width +
                                  hi * width + wi] *
                              weight[channel * in_channels * kD * kH * kW +
                                     c * kD * kH * kW +
                                     kd * kH * kW + kh * kW + kw];
                    }
                }
            }
        }
        output[output_idx] = val;
    }
}

std::tuple<torch::Tensor> conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto depth = input.size(2);
    const auto height = input.size(3);
    const auto width = input.size(4);

    const auto out_channels = weight.size(0);
    const auto kD = weight.size(2);
    const auto kH = weight.size(3);
    const auto kW = weight.size(4);

    // Output dimensions calculation
    const int out_depth = (depth + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;
    const int out_height = (height + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    const int out_width = (width + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    const int num_kernels = batch_size * out_channels * out_depth * out_height * out_width;

    dim3 blocks((num_kernels + 512 - 1) / 512);
    dim3 threads(512);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, depth, height, width,
            out_channels, kD, kH, kW,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_cpp_source = "std::tuple<torch::Tensor> conv3d_forward(torch::Tensor input, torch::Tensor weight, int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w, int dilation_d, int dilation_h, int dilation_w);"

custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_kernel,
    functions=["conv3d_forward"],
    verbose=True,
    with_cuda=True,
    extra_cflags=["-g", "-DCUDA_HAS_FP16=1"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights with same parameters as PyTorch's Conv3d
        weight_shape = (out_channels, in_channels // groups, *kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights like PyTorch's default initializer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare convolution parameters
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation
        
        output = custom_conv3d.conv3d_forward(
            x.contiguous(),
            self.weight.contiguous(),
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output