import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the CUDA kernel for optimized 4D tensor-matrix multiplication
        optimized_einsum_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void optimized_einsum_kernel(
            const scalar_t* __restrict__ A,
            const scalar_t* __restrict__ B,
            scalar_t* __restrict__ C,
            const int b,
            const int i,
            const int j,
            const int l,
            const int k
        ) {
            int batch_idx = blockIdx.z;
            int i_idx = blockIdx.y;
            int j_idx = blockIdx.x;
            int k_idx = threadIdx.x;

            scalar_t sum = 0;
            for (int l_idx = 0; l_idx < l; ++l_idx) {
                sum += A[batch_idx * i * j * l + i_idx * j * l + j_idx * l + l_idx] *
                       B[l_idx * k + k_idx];
            }

            C[batch_idx * i * j * k + i_idx * j * k + j_idx * k + k_idx] = sum;
        }

        torch::Tensor optimized_einsum_cuda(torch::Tensor A, torch::Tensor B) {
            const int b = A.size(0);
            const int i = A.size(1);
            const int j = A.size(2);
            const int l = A.size(3);
            const int k = B.size(1);

            auto C = torch::zeros({b, i, j, k}, A.options());

            dim3 threads_per_block(std::min(k, 256));
            dim3 blocks_per_grid(j, i, b);

            AT_DISPATCH_FLOATING_TYPES(A.type(), "optimized_einsum_cuda", ([&] {
                optimized_einsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    C.data_ptr<scalar_t>(),
                    b, i, j, l, k
                );
            }));

            return C;
        }
        """

        optimized_einsum_cpp_source = (
            "torch::Tensor optimized_einsum_cuda(torch::Tensor A, torch::Tensor B);"
        )

        # Load the CUDA kernel
        self.optimized_einsum = load_inline(
            name="optimized_einsum",
            cpp_sources=optimized_einsum_cpp_source,
            cuda_sources=optimized_einsum_source,
            functions=["optimized_einsum_cuda"],
            verbose=True
        )

    def forward(self, A, B):
        return self.optimized_einsum.optimized_einsum_cuda(A, B)