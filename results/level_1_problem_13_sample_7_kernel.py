import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for symmetric matrix multiplication
symmetric_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void symmetric_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int N
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;

    __shared__ float shared_a[2][32 * 32];  // Assuming block size 32 for better cache
    __shared__ float shared_b[2][32 * 32];  

    int row = bid * block_size + tid;
    int col = row;

    float sum = 0.0;

    for (int k = 0; k < N; k += block_size * 2) {
        int a_row = row;
        int b_col = col;
        
        // Load A and B tiles into shared memory
        // Read A's row
        if (tid < N) {
            shared_a[0][tid] = a[row * N + k + tid];
            shared_a[1][tid] = a[row * N + (N - 1 - (k + tid))];
        }
        // Read B's column
        if (tid < N) {
            shared_b[0][tid] = b[(k + tid) * N + col];
            shared_b[1][tid] = b[(N - 1 - (k + tid)) * N + col];
        }
        __syncthreads();

        // Compute the block of the current tile
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                sum += shared_a[0][i * block_size + tid] * shared_b[0][j * block_size + tid];
            }
        }
        __syncthreads();
    }

    c[row * N + col] = sum;
    if (row != col) {
        c[col * N + row] = sum;
    }
}

torch::Tensor symmetric_matmul_cuda(
    torch::Tensor a,
    torch::Tensor b,
    int N
) {
    auto output = torch::empty({N, N}, a.options());

    const int block_size = 32;
    const dim3 grid(N); // Because each block handles a row and column pair
    const dim3 threads(block_size);

    symmetric_matmul_kernel<<<grid, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), N);

    return output;
}
"""

symmetric_matmul_cpp_source = "torch::Tensor symmetric_matmul_cuda(torch::Tensor a, torch::Tensor b, int N);"

symmetric_matmul = load_inline(
    name="symmetric_matmul",
    cpp_sources=symmetric_matmul_cpp_source,
    cuda_sources=symmetric_matmul_source,
    functions=["symmetric_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.N = N  # Assuming N is defined in global scope
        self.symmetric_matmul = symmetric_matmul.symmetric_matmul_cuda

    def forward(self, A, B):
        return self.symmetric_matmul(A, self.N) @ B  # Or whichever arrangement respects the symmetry
        # Note: The kernel might need adjustments to correctly handle both A and B's symmetry
        # This is a simplified representation and may require further kernel optimization