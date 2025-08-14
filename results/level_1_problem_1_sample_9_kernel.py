import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
        # Define and load the custom CUDA kernel for matrix multiplication
        matrix_mul_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < N && col < N) {
                float sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }

        torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B, int N) {
            const int block_size = 32;
            dim3 block(block_size, block_size);
            dim3 grid((N + block.x - 1)/block.x, (N + block.y - 1)/block.y);

            auto C = torch::empty({N, N}, torch::kFloat32).cuda();

            matrix_mul_kernel<<<grid, block>>>(
                A.data_ptr<float>(), 
                B.data_ptr<float>(), 
                C.data_ptr<float>(), 
                N
            );

            return C;
        }
        """

        matrix_mul_cpp_source = "torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B, int N);"

        # Compile and load the CUDA kernel
        self.matrix_mul = load_inline(
            name="matrix_mul",
            cpp_sources=matrix_mul_cpp_source,
            cuda_sources=matrix_mul_source,
            functions=["matrix_mul_cuda"],
            verbose=True,
        )
        
        self.N = 2048 * 2  # Save N as class variable

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mul.matrix_mul_cuda(A.cuda(), B.cuda(), self.N)