import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Define and compile the custom CUDA kernel for batch matrix multiplication
        batched_matmul_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cuda_fp16.h>

        __global__ void batched_matmul_kernel(const float* A, const float* B, float* C,
                                             int batch_size, int m, int k, int n) {
            int batch = blockIdx.z;
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m && col < n) {
                float sum = 0.0;
                for (int i = 0; i < k; ++i) {
                    sum += A[batch * m * k + row * k + i] * B[batch * k * n + i * n + col];
                }
                C[batch * m * n + row * n + col] = sum;
            }
        }

        torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
            int batch_size = A.size(0);
            int m = A.size(1);
            int k = A.size(2);
            int n = B.size(2);

            auto C = torch::empty({batch_size, m, n}, A.options());

            dim3 threads(32, 8);
            dim3 blocks((n + threads.x - 1)/threads.x, (m + threads.y - 1)/threads.y, batch_size);
            
            batched_matmul_kernel<<<blocks, threads>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                batch_size, m, k, n
            );

            return C;
        }
        """

        batched_matmul_cpp_source = "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"

        self.batched_matmul = load_inline(
            name="batched_matmul",
            cpp_sources=batched_matmul_cpp_source,
            cuda_sources=batched_matmul_source,
            functions=["batched_matmul_cuda"],
            verbose=True
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)