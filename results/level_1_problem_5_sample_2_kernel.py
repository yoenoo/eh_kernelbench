import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scalar_mul_kernel(const float* A, float s, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor scalar_mul_cuda(torch::Tensor A, float s) {
    int64_t size = A.numel();
    auto options = torch::TensorOptions().like(A);
    torch::Tensor C = torch::empty({size}, options);
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    scalar_mul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), size);
    
    return C.view_as(A);
}
"""

scalar_mul_cpp_src = "torch::Tensor scalar_mul_cuda(torch::Tensor A, float s);"

scalar_mul = load_inline(
    name="scalar_mul",
    cpp_sources=scalar_mul_cpp_src,
    cuda_sources=scalar_source,
    functions=["scalar_mul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scalar_mul = scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mul.scalar_mul_cuda(A, s)