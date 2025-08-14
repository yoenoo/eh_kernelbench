import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
matmul_scalar_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_scalar_kernel(const float* a, float s, float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        out[idx] = a[idx] * s;
    }
}

torch::Tensor matmul_scalar_cuda(torch::Tensor a, float s) {
    int rows = a.size(0);
    int cols = a.size(1);
    auto out = torch::empty({rows, cols}, a.options());

    int elements = rows * cols;
    const int block_size = 256;
    const int num_blocks = (elements + block_size - 1) / block_size;

    matmul_scalar_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), s, out.data_ptr<float>(), rows, cols);

    return out;
}
"""

matmul_scalar_cpp_source = (
    "torch::Tensor matmul_scalar_cuda(torch::Tensor a, float s);"
)

# Compile the inline CUDA code for matrix-scalar multiplication
matmul_scalar = load_inline(
    name="matmul_scalar",
    cpp_sources=matmul_scalar_cpp_source,
    cuda_sources=matmul_scalar_source,
    functions=["matmul_scalar_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_scalar = matmul_scalar

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.matmul_scalar.matmul_scalar_cuda(A, s)