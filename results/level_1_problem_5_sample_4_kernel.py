import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Define and load the CUDA kernel inline
        scalar_mult_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void scalar_mult_kernel(const float* A, float s, float* C, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                C[idx] = A[idx] * s;
            }
        }

        torch::Tensor scalar_mult_cuda(torch::Tensor A, float s) {
            int size = A.numel();
            auto C = torch::empty_like(A);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            scalar_mult_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), size);
            cudaDeviceSynchronize();  // Ensure the kernel has finished

            return C;
        }
        """
        # Load the CUDA extension
        self.scalar_mult = load_inline(
            name='scalar_mult',
            cpp_sources="torch::Tensor scalar_mult_cuda(torch::Tensor A, float s);",
            cuda_sources=scalar_mult_source,
            functions=['scalar_mult_cuda'],
            verbose=True
        )

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mult.scalar_mult_cuda(A.cuda(), s).cpu() if A.is_cpu else self.scalar_mult.scalar_mult_cuda(A, s)