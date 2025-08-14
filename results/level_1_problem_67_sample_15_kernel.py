import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv1D
conv1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    const int batch_size, const int in_channels, const int input_length,
    const int out_channels, const int kernel_size,
    const int stride, const int padding, const int dilation) {

    const int B = blockIdx.z;  // batch index
    const int C = blockIdx.y;  // output channel index
    const int K = threadIdx.x; // kernel index

    const int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    const int output_length = (input_length + 2 * padding - effective_kernel_size) / stride + 1;

    for (int O = blockIdx.x; O < output_length; O += gridDim.x) {
        scalar_t sum = 0;
        for (int D = 0; D < dilation; D++) {
            const int I = O * stride - padding + D * dilation + K;
            if (I >=0 && I < input_length) {
                for (int G = 0; G < in_channels; G++) { // Assuming groups=1
                    sum += input[B][G][I] * weight[C][G][K];
                }
            }
        }
        output[B][C][O] = sum;
    }
}

at::Tensor conv1d_forward_cuda(const at::Tensor& input, const at::Tensor& weight,
    int stride, int padding, int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Compute output dimensions
    const int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    const int output_length = (input_length + 2 * padding - effective_kernel_size) / stride + 1;

    // Define output tensor
    auto output = at::empty({batch_size, out_channels, output_length}, input.options());

    // Define grid and block dimensions
    const dim3 threads(kernel_size);
    const dim3 blocks(output_length < 1 ? 1 : output_length,
        out_channels,
        batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv1d_forward_cuda", ([&] {
        conv1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            batch_size, in_channels, input_length,
            out_channels, kernel_size,
            stride, padding, dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_header = """
at::Tensor conv1d_forward_cuda(const at::Tensor& input, const at::Tensor& weight,
    int stride, int padding, int dilation);
"""

conv1d_cuda = load_inline(
    name="conv1d_cuda",
    cpp_sources=conv1d_header,
    cuda_sources=conv1d_source,
    functions="conv1d_forward_cuda",
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # Assuming groups=1 for simplicity in kernel
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = conv1d_cuda.conv1d_forward_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output