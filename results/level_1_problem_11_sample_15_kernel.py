import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 4D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void tensor_matmul_cuda_kernel(const scalar_t* __restrict__ A,
                                       const scalar_t* __restrict__ B,
                                       scalar_t* __restrict__ C,
                                       const int b, const int i, const int j,
                                       const int l, const int k) {
    int batch_idx = blockIdx.x;
    int i_idx = blockIdx.y;
    int j_idx = blockIdx.z;
    int tid = threadIdx.x;

    scalar_t sum = 0.0;
    for (int l_idx = tid; l_idx < l; l_idx += blockDim.x) {
        sum += A[batch_idx * i * j * l + i_idx * j * l + j_idx * l + l_idx] *
               B[l_idx * k + tid / (blockDim.x / l)];
    }

    // Synchronize to ensure all thread computations are done
    __syncthreads();

    // Perform final summation using parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            C[batch_idx * i * j * k + i_idx * j * k + j_idx * k + k_idx] = sum;
        }
    }
}

template <typename scalar_t>
torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int b = A.size(0);
    const int i = A.size(1);
    const int j = A.size(2);
    const int l = A.size(3);
    const int k = B.size(1);

    torch::Tensor C = torch::zeros({b, i, j, k}, A.options());

    const int block_size = 256;
    dim3 grid(b, i, j);
    dim3 block(block_size);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "tensor_matmul_cuda", ([&] {
        tensor_matmul_cuda_kernel<scalar_t><<<grid, block>>>(
            A.data<scalar_t>(), B.data<scalar_t>(),
            C.data<scalar_t>(), b, i, j, l, k);
    }));

    cudaDeviceSynchronize();
    return C;
}

torch::Tensor tensor_matmul_forward(torch::Tensor A, torch::Tensor B) {
    return tensor_matmul_cuda(A, B);
}
"""

tensor_matmul_cpp_source = """
#include <torch/extension.h>
torch::Tensor tensor_matmul_forward(torch::Tensor A, torch::Tensor B);
"""

tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_forward"],
    verbose=True,
    extra_cflags=["-g"],
    extra_cuda_cflags=["-g", "-O3"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_matmul = tensor_matmul

    def forward(self, A, B):
        return self.custom_matmul.tensor_matmul_forward(A, B).cuda()