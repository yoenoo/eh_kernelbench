import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Compile the batched matrix multiplication CUDA kernel
        self.bmm_kernel = load_inline(
            name='bmm_cuda',
            cpp_sources=r"""
            torch::Tensor batched_matrix_mul_cuda(torch::Tensor A, torch::Tensor B);
            """,
            cuda_sources=r"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            #include <cuda_fp16.h>

            template <typename T>
            __global__ void batched_matrix_mul_kernel(const T* A, const T* B, T* C, int batch_size, int m, int k, int n) {
                int batch = blockIdx.z;
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row < m && col < n) {
                    T sum = 0;
                    for (int e = 0; e < k; e++) {
                        sum += A[batch * m * k + row * k + e] * B[batch * k * n + e * n + col];
                    }
                    C[batch * m * n + row * n + col] = sum;
                }
            }

            torch::Tensor batched_matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
                const int batch_size = A.size(0);
                const int m = A.size(1);
                const int k = A.size(2);
                const int n = B.size(2);

                auto C = torch::empty({batch_size, m, n}, A.options());

                dim3 threads(32, 8);
                dim3 blocks((n + threads.x - 1)/threads.x, (m + threads.y - 1)/threads.y, batch_size);

                AT_DISPATCH_ALL_TYPES(A.type(), "batched_matrix_mul_cuda", ([&] {
                    batched_matrix_mul_kernel<scalar_t><<<blocks, threads>>>(
                        A.data_ptr<scalar_t>(), 
                        B.data_ptr<scalar_t>(), 
                        C.data_ptr<scalar_t>(),
                        batch_size, m, k, n);
                }));

                return C;
            }
            """,
            functions=['batched_matrix_mul_cuda'],
            verbose=False
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device
        A = A.cuda()
        B = B.cuda()
        return self.bmm_kernel.batched_matrix_mul_cuda(A, B)