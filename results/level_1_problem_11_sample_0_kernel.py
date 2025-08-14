import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define and compile the custom CUDA kernel
        kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void tensor_matmul_kernel(
            const scalar_t* __restrict__ A,
            const scalar_t* __restrict__ B,
            scalar_t* __restrict__ C,
            const int B_dim,
            const int I_dim,
            const int J_dim,
            const int L_dim,
            const int K_dim
        ) {
            int b = blockIdx.x;
            int i = blockIdx.y;
            int j = blockIdx.z;
            int k = threadIdx.x;

            scalar_t sum = 0.0;
            for (int l = 0; l < L_dim; ++l) {
                sum += A[b * I_dim * J_dim * L_dim + i * J_dim * L_dim + j * L_dim + l] *
                       B[l * K_dim + k];
            }
            C[b * I_dim * J_dim * K_dim + i * J_dim * K_dim + j * K_dim + k] = sum;
        }

        torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
            const int B_dim = A.size(0);
            const int I_dim = A.size(1);
            const int J_dim = A.size(2);
            const int L_dim = A.size(3);
            const int K_dim = B.size(1);

            auto C = torch::empty({B_dim, I_dim, J_dim, K_dim}, A.options());

            dim3 threads(K_dim);
            dim3 blocks(B_dim, I_dim, J_dim);

            AT_DISPATCH_FLOATING_TYPES(A.type(), "tensor_matmul_cuda", ([&] {
                tensor_matmul_kernel<scalar_t><<<blocks, threads>>>(
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    C.data_ptr<scalar_t>(),
                    B_dim, I_dim, J_dim, L_dim, K_dim
                );
            }));

            return C;
        }
        """

        # Compile the CUDA kernel inline
        self.custom_tensor_matmul = load_inline(
            name='tensor_matmul',
            cpp Sources='',
            cuda_sources=kernel_source,
            functions=['tensor_matmul_cuda'],
            verbose=True
        )

    def forward(self, A, B):
        return self.custom_tensor_matmul.tensor_matmul_cuda(A, B)