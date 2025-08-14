import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def depthwise_conv2d_kernel_backward(grad_output, input, weight, stride=1, padding=0):
    # This function would contain the backward pass implementation for the custom depthwise Conv2d
    raise NotImplementedError("Backward pass not implemented for custom depthwise_conv2d.")

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel_forward(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int stride, const int padding) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_size = weight.size(2); // assuming square kernel

    const int output_height = output.size(2);
    const int output_width = output.size(3);

    const int H_out = output_height;
    const int W_out = output_width;

    const int N = blockIdx.x;
    const int C = blockIdx.y;
    const int H = blockIdx.z;
    const int W = threadIdx.x;

    if (H >= H_out || W >= W_out) return;

    const int H_in = H * stride - padding;
    const int W_in = W * stride - padding;

    scalar_t val = 0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            const int h_k = H_in + ky;
            const int w_k = W_in + kx;
            if (h_k >= 0 && h_k < input_height && w_k >=0 && w_k < input_width) {
                val += input[N][C][h_k][w_k] * weight[C][0][ky][kx];
            }
        }
    }
    output[N][C][H][W] = val;
}

torch::Tensor depthwise_conv2d_cuda_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto kernel_size = weight.size(2);

    // Calculate output dimensions
    auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(32);
    dim3 blocks(batch_size, channels, output_height);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&]{
        depthwise_conv2d_kernel_forward<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            stride, padding);
    }));

    return output;
}

// Backward function will need similar CUDA kernels for grad_input and grad_weight but omitted here for brevity

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthwise_conv2d_cuda_forward, "Depthwise Conv2d forward CUDA");
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor forward(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

depthwise_conv2d_cuda = load_inline(
    name="depthwise_conv2d_cuda",
    cpp_sources=cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ModelNew, self).__init__()
        assert in_channels == out_channels, "Depthwise requires in_channels == out_channels"
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return depthwise_conv2d_cuda.forward(x, self.weight, self.stride, self.padding)