import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void tanh_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = tanhf(x[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    int n = x.numel();
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor y = torch::empty(n, options);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size, 0, torch::cuda::current_stream()>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y.view_as(x);
}
"""

tanh_cpp_header = "torch::Tensor tanh_cuda(torch::Tensor x);"

tanh_extension = load_inline(
    name="tanh_extension",
    cpp_sources=[tanh_cpp_header],
    cuda_sources=[tanh_cuda_source],
    functions=["tanh_cuda"],
    verbose=True,
    extra_cflags=["-D護"],
    extra_cuda_cflags=["-lineinfo", "-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_cuda = tanh_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda.tanh_cuda(x)