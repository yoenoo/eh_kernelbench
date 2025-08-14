import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> bias,
                                    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                    const int batch_size, const int in_channels,
                                    const int out_channels, const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w,
                                    const int pad_h, const int pad_w,
                                    const int dilation_h, const int dilation_w) {

    const int H_out = output.size(2);
    const int W_out = output.size(3);
    const int num_kernels = batch_size * out_channels * H_out * W_out;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int w = idx % W_out;
    const int h = (idx / W_out) % H_out;
    const int c_out = (idx / W_out / H_out) % out_channels;
    const int n = idx / (out_channels * H_out * W_out);

    scalar_t sum = bias[c_out];
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = -pad_h + h * stride_h + kh * dilation_h;
                int w_in = -pad_w + w * stride_w + kw * dilation_w;
                if (h_in >= 0 && h_in < input.size(2) && w_in >=0 && w_in < input.size(3)) {
                    sum += weight[c_out][c_in][kh][kw] * input[n][c_in][h_in][w_in];
                }
            }
        }
    }
    output[n][c_out][h][w] = sum;
}

torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w, int dilation_h, int dilation_w) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);

    // Compute output dimensions
    auto H = input.size(2);
    auto W = input.size(3);
    auto H_out = (H + 2*pad_h - dilation_h*(kernel_h-1) -1)/stride_h + 1;
    auto W_out = (W + 2*pad_w - dilation_w*(kernel_w-1) -1)/stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * out_channels * H_out * W_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv2d", ([&] {
        custom_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w, int dilation_h, int dilation_w);
"""

conv2d_extension = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["custom_conv2d"],
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
        
        # Initialize weights and bias similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Compile CUDA extension
        self.custom_conv2d = conv2d_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation

        # Handle groups (simplified for this example, assumes groups=1)
        if self.groups != 1:
            raise NotImplementedError("Groups >1 not supported in current implementation")
            
        # Execute custom CUDA kernel
        if self.bias is not None:
            return self.custom_conv2d(
                x, self.weight, self.bias,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w
            )
        else:
            # If no bias, pass zeros tensor
            zeros_bias = torch.zeros(self.out_channels, device=x.device)
            return self.custom_conv2d(
                x, self.weight, zeros_bias,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w
            )