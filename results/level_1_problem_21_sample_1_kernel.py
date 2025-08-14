import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        scalar_t z = input[tid];
        output[tid] = 1.0 / (1.0 + exp(-z));
    }
}

void sigmoid_launch(torch::Tensor input, torch::Tensor output, int size) {
    const int block_size = 512;
    const int grid_size = (size + block_size - 1) / block_size;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "sigmoid_kernel", ([&] {
        sigmoid_kernel<scalar_t><<<grid_size, block_size>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), size);
    }));
    cudaDeviceSynchronize();
}
"""

sigmoid_cpp_source = """
#include <torch/extension.h>

void sigmoid_launch(torch::Tensor input, torch::Tensor output, int size);
torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    sigmoid_launch(input, output, input.numel());
    return output;
}
"""

# Compile the CUDA code
sigmoid_cuda = load_inline(
    name="sigmoid_cuda",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_kernel_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA", "-x cu", "-arch=sm_86"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.sigmoid_cuda(x)