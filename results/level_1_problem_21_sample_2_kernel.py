import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void sigmoid_forward_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        scalar_t x = input[index];
        scalar_t inv_exp_x = exp(-x);
        output[index] = 1.0 / (1.0 + inv_exp_x);
    }
}

template <typename scalar_t>
void sigmoid_forward_cuda(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    sigmoid_forward_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), n);
}

void sigmoid_forward(torch::Tensor input, torch::Tensor output) {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "sigmoid_forward", ([&] {
        sigmoid_forward_cuda<scalar_t>(input, output);
    }));
}
"""

sigmoid_kernel_header = """
void sigmoid_forward(torch::Tensor input, torch::Tensor output);
"""

sigmoid = load_inline(
    name="sigmoid_cuda",
    cpp_sources=sigmoid_kernel_header,
    cuda_sources=sigmoid_kernel_source,
    functions=["sigmoid_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_forward = sigmoid.sigmoid_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        self.sigmoid_forward(x, output)
        return output