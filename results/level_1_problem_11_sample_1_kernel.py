import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tensor_matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void tensor_matrix_mul(const scalar_t* __restrict__ A, 
                                 const scalar_t* __restrict__ B, 
                                 scalar_t* __restrict__ C,
                                 int b, int i, int j, int l, int k) {
    int batch_idx = blockIdx.z;
    int output_i = blockIdx.x * blockDim.y + threadIdx.y;
    int output_j = blockIdx.y * blockDim.x + threadIdx.x;
    if (output_i >= i || output_j >= j) return;

    scalar_t sum = 0;
    for (int p = 0; p < l; ++p) {
        sum += A[batch_idx * i * j * l + output_i * j * l + output_j * l + p] * 
               B[p * k + threadIdx.z];
    }
    C[batch_idx * i * j * k + output_i * j * k + output_j * k + threadIdx.z] = sum;
}

at::Tensor tensor_matrix_mul_cuda(at::Tensor A, at::Tensor B) {
    const int b = A.size(0);
    const int i = A.size(1);
    const int j = A.size(2);
    const int l = A.size(3);
    const int k = B.size(1);

    at::Tensor C = at::empty({b, i, j, k}, A.options());

    const int block_x = 32;
    const int block_y = 8;
    const dim3 threads(block_x, block_y, 1);
    dim3 blocks(j / block_x, i / block_y, b);

    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "tensor_matrix_mul_cuda", ([&] {
        tensor_matrix_mul<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), 
            C.data_ptr<scalar_t>(), b, i, j, l, k);
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

cpp_source = "at::Tensor tensor_matrix_mul_cuda(at::Tensor A, at::Tensor B);"

tensor_matrix_mult = load_inline(
    name="tensor_matrix_mult",
    cpp_sources=[cpp_source],
    cuda_sources=[tensor_matrix_mult_source],
    functions=["tensor_matrix_mul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_mult = tensor_matrix_mult

    def forward(self, A, B):
        return self.custom_mult.tensor_matrix_mul_cuda(A, B)