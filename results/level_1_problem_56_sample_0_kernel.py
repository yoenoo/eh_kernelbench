import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution with asymmetric kernel and padding
conv2d_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// Function to calculate the output dimensions
static inline int64_t output_size(int64_t input_size, int64_t filter_size, int64_t padding, int64_t stride, int64_t dilation) {
  return (input_size + 2 * padding - dilation * (filter_size - 1) - 1) / stride + 1;
}

__global__ void asymmetric_conv2d_forward(
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {

    int batch = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_h = threadIdx.x;
    int out_w = threadIdx.y;

    float val = 0;

    int in_channels_per_group = input.size(1) / groups;
    int out_channels_per_group = weight.size(0) / groups;
    int group_id = out_channel / out_channels_per_group;

    int in_channel_start = group_id * in_channels_per_group;
    int in_channel_end = in_channel_start + in_channels_per_group;

    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            int h = out_h * stride_h - padding_h + dilation_h * i;
            int w = out_w * stride_w - padding_w + dilation_w * j;
            if (h >= 0 && h < input.size(2) && w >= 0 && w < input.size(3)) {
                for (int c = in_channel_start; c < in_channel_end; c++) {
                    val += input[batch][c][h][w] * weight[out_channel][c - in_channel_start][i][j];
                }
            }
        }
    }

    output[batch][out_channel][out_h][out_w] = val;
}

torch::Tensor asymmetric_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {

    // Get input and output dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_channels = weight.size(0);
    int out_h = output_size(in_h, kernel_h, padding_h, stride_h, dilation_h);
    int out_w = output_size(in_w, kernel_w, padding_w, stride_w, dilation_w);

    torch::Tensor output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    // Define grid and block dimensions
    dim3 threads(out_h, out_w); // each thread handles one output position
    dim3 grid(batch_size, out_channels);

    // Launch the kernel
    asymmetric_conv2d_forward<<<grid, threads>>>(
        input.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor asymmetric_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups);
"""

# Load the CUDA kernel
conv2d_op = load_inline(
    name="asymmetric_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel,
    functions=["asymmetric_conv2d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.asymmetric_conv2d_forward_cuda = conv2d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.asymmetric_conv2d_forward_cuda.asymmetric_conv2d_forward_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out