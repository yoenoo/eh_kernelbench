import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom 1D convolution CUDA kernel
conv1d_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_forward(const scalar_t* __restrict__ input,
                              const scalar_t* __restrict__ weight,
                              scalar_t* __restrict__ output,
                              int in_channels,
                              int out_channels,
                              int kernel_size,
                              int input_length,
                              int output_length,
                              int stride,
                              int padding,
                              int dilation) {

    const int batch_id = blockIdx.x;
    const int out_ch = blockIdx.y;
    const int in_ch = threadIdx.z;
    
    // Each thread computes one output element
    for (int pos = threadIdx.x + blockIdx.z * blockDim.x;
         pos < output_length;
         pos += gridDim.z * blockDim.x) {

        scalar_t sum = 0;
        for (int kk = 0; kk < kernel_size; ++kk) {
            // Compute input position
            int input_pos = pos * stride - padding + kk * dilation;
            if (input_pos < 0 || input_pos >= input_length) continue;

            sum += weight[out_ch * in_channels * kernel_size + in_ch * kernel_size + kk] *
                   input[batch_id * in_channels * input_length + in_ch * input_length + input_pos];
        }
        atomicAdd(&output[batch_id * out_channels * output_length + out_ch * output_length + pos], sum);
    }
}

torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Compute output length
    int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    int threads_per_block = 256;
    dim3 blocks(batch_size, out_channels, 1);
    dim3 threads(threads_per_block, 1, in_channels);

    conv1d_forward<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        in_channels, out_channels, kernel_size,
        input_length, output_length,
        stride, padding, dilation);

    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_cpp_source = (
    "torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

conv1d_op = load_inline(
    name="conv1d_op",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_kernel,
    functions=["conv1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        assert groups == 1, "Groups > 1 not supported in current kernel implementation"
        
        # Initialize weights similar to PyTorch's Conv1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        # Register custom op
        self.conv1d_op = conv1d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1d_op.conv1d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out