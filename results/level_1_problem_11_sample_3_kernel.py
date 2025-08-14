import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

        # Define the custom CUDA kernel for 4D tensor-matrix multiplication
        tensor_matmul_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void tensor_matrix_mul_kernel(
            const scalar_t* __restrict__ A,
            const scalar_t* __restrict__ B,
            scalar_t* __restrict__ C,
            const int B_dim,
            const int I_dim,
            const int J_dim,
            const int L_dim,
            const int K_dim
        ) {
            const int b = blockIdx.x;
            const int i = blockIdx.y;
            const int j = threadIdx.x;
            const int k = threadIdx.y;

            scalar_t sum = 0.0;
            for (int l = 0; l < L_dim; ++l) {
                sum += A[b * I_dim * J_dim * L_dim + i * J_dim * L_dim + j * L_dim + l] * 
                       B[l * K_dim + k];
            }
            C[b * I_dim * J_dim * K_dim + i * J_dim * K_dim + j * K_dim + k] = sum;
        }

        torch::Tensor tensor_matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
            const int B_dim = A.size(0);
            const int I_dim = A.size(1);
            const int J_dim = A.size(2);
            const int L_dim = A.size(3);
            const int K_dim = B.size(1);

            auto C = torch::empty({B_dim, I_dim, J_dim, K_dim}, A.options());

            dim3 threads(J_dim, K_dim);
            dim3 blocks(B_dim, I_dim);

            AT_DISPATCH_FLOATING_TYPES(A.type(), "tensor_matrix_mul_cuda", ([&] {
                tensor_matrix_mul_kernel<scalar_t><<<blocks, threads>>>(
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    C.data_ptr<scalar_t>(),
                    B_dim, I_dim, J_dim, L_dim, K_dim);
            }));

            return C;
        }
        """

        tensor_matmul_cpp_source = """
        torch::Tensor tensor_matrix_mul_cuda(torch::Tensor A, torch::Tensor B);
        """

        # Compile the inline CUDA code
        self.tensor_matmul = load_inline(
            name="tensor_matrix_mul",
            cpp_sources=tensor_matmul_cpp_source,
            cuda_sources=tensor_matmul_source,
            functions=["tensor_matrix_mul_cuda"],
            verbose=True
        )

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matrix_mul_cuda(A, B)