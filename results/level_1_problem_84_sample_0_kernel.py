import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int kernel_size, int stride,
    int height_in, int width_in, int height_out, int width_out) {

    const int H_out = output.size(2);
    const int W_out = output.size(3);
    const int KW = kernel_size;
    const int KH = kernel_size;

    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;

    // output feature map index
    for (int hi = h; hi < H_out; hi += blockDim.y) {
        for (int wi = w; wi < W_out; wi += blockDim.x) {
            scalar_t sum = 0;
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int h_in = hi * stride + kh;
                    int w_in = wi * stride + kw;
                    if (h_in < height_in && w_in < width_in) {
                        sum += input[n][c][h_in][w_in] * weight[c][0][kh][kw];
                    }
                }
            }
            output[n][c][hi][wi] = sum;
        }
    }
}

std::vector<torch::Tensor> depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int kernel_size = weight.size(2);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int out_channels = weight.size(0);

    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const dim3 threads(32, 8);
    dim3 blocks(batch_size, out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, kernel_size, stride,
            height_in, width_in, height_out, width_out);
    }));

    cudaDeviceSynchronize();
    return {output};
}

"""

# Compile the CUDA kernel
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources="",
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x):
        outputs = self.depthwise_conv2d.depthwise_conv2d_forward(
            x, 
            self.weight, 
            self.stride, 
            self.padding
        )[0]

        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1, 1)

        return outputs