import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_conv_kernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     int batch_size,
                                     int in_channels,
                                     int out_channels,
                                     int height,
                                     int width) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batch_size * height * width) return;

    int batch = n / (height * width);
    int pos = n % (height * width);

    for (int oc = 0; oc < out_channels; oc++) {
        scalar_t sum = 0;
        for (int ic = 0; ic < in_channels; ic++) {
            const scalar_t in_val = input[batch * in_channels * height * width + ic * height * width + pos];
            const scalar_t wt_val = weight[oc * in_channels + ic];
            sum += in_val * wt_val;
        }
        output[batch * out_channels * height * width + oc * height * width + pos] = sum;
    }
}

std::vector<torch::Tensor> pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor output) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height = input.size(2);
    const int width = input.size(3);

    const int total_elements = batch_size * height * width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    const int block_size = 256;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv_cuda", ([&] {
        pointwise_conv_kernel<scalar_t><<<blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width);
    }));

    cudaDeviceSynchronize();
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pointwise_conv", &pointwise_conv_cuda, "Pointwise Convolution CUDA");
}
"""

pointwise_conv_cpp = """
#include <torch/extension.h>
std::vector<torch::Tensor> pointwise_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor output);
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.bias_term = bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.pointwise_conv = load_inline(
            name="pointwise_conv",
            cpp_sources=pointwise_cpp,
            cuda_sources=pointwise_conv_source,
            functions=["pointwise_conv"],
            verbose=True,
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape
        output = torch.empty((batch_size, self.weight.size(0), height, width), device=x.device)
        out = self.pointwise_conv.pointwise_conv(x, self.weight.view(self.weight.size(0), -1), output)[0]
        if self.bias_term:
            out += self.bias.view(1, -1, 1, 1)
        return out