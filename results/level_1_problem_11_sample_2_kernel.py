import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 4D tensor-matrix multiplication
tensor_matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tensor_matrix_mult_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int b, const int i, const int j, const int l, const int k) {

    int batch_idx = blockIdx.z;
    int i_idx = blockIdx.y;
    int j_idx = blockIdx.x;
    int k_idx = threadIdx.x;

    scalar_t sum = 0.0;
    for (int l_idx = 0; l_idx < l; ++l_idx) {
        sum += A[batch_idx * i * j * l + i_idx * j * l + j_idx * l + l_idx] *
               B[l_idx * k + k_idx];
    }

    C[batch_idx * i * j * k + i_idx * j * k + j_idx * k + k_idx] = sum;
}

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int b = A.size(0);
    const int i = A.size(1);
    const int j = A.size(2);
    const int l = A.size(3);
    const int k = B.size(1);

    auto C = torch::empty({b, i, j, k}, A.options());

    dim3 threads(256);
    dim3 blocks(j, i, b);
    const int k_size = k;
    if (k_size < 256) {
        threads.x = k_size;
    }

    AT_DISPATCH_FLOATING_TYPES(A.type(), "tensor_matrix_mult_cuda", ([&] {
        tensor_matrix_mult_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            C.data<scalar_t>(),
            b, i, j, l, k);
    }));

    return C;
}
"""

tensor_matrix_mult_cpp_source = """
torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for 4D tensor-matrix multiplication
tensor_matrix_mult = load_inline(
    name="tensor_matrix_mult",
    cpp_sources=tensor_matrix_mult_cpp_source,
    cuda_sources=tensor_matrix_mult_source,
    functions=["tensor_matrix_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matrix_mult = tensor_matrix_mult

    def forward(self, A, B):
        return self.tensor_matrix_mult.tensor_matrix_mult_cuda(A, B)