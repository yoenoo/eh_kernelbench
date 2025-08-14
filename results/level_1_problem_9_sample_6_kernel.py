import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tall matrix multiplication
matmul_tall_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

template <typename T>
__global__ void matmul_kernel(
    const T* __restrict__ A, 
    const T* __restrict__ B, 
    T* C, 
    int m, 
    int k, 
    int n
) {
    const int warpSize = 32;
    const int warpRows = 4;
    const int blockRows = 8;
    const int colsPerThread = 4;
    
    extern __shared__ T shared[];
    T* sharedA = shared;
    T* sharedB = sharedA + blockDim.x * warpRows;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * (colsPerThread * blockDim.x) + tx * colsPerThread;

    T* c = C + row * n + col;

    T sum[colsPerThread] = {0};

    for (int batch = 0; batch < (k + warpSize * warpRows - 1) / (warpSize * warpRows); ++batch) {
        int aCol = batch * warpSize * warpRows + tx * warpRows + ty / warpSize;
        aCol = min(aCol, k - 1);

        // Load A tiles
        for (int r = 0; r < warpRows; ++r) {
            int aRow = by * blockDim.y * blockRows + ty % warpSize * blockRows + r;
            if (aRow < m && aCol < k) {
                sharedA[ty * warpRows + r] = A[aRow * k + aCol];
            } else {
                sharedA[ty * warpRows + r] = 0;
            }
        }

        // Load B tiles
        for (int c = 0; col + c < n && batch * warpSize * warpRows + tx * warpRows + (ty % warpSize) < k; ++c) {
            int bRow = batch * warpSize * warpRows + tx * warpRows + (ty % warpSize);
            int bCol = col + c;
            if (bRow < k && bCol < n) {
                sharedB[ty * colsPerThread + c] = B[bRow * n + bCol];
            } else {
                sharedB[ty * colsPerThread + c] = 0;
            }
        }

        __syncthreads();

        // Compute dot product
        for (int r = 0; r < warpRows; ++r) {
            for (int c = 0; c < colsPerThread; ++c) {
                sum[c] += sharedA[ty * warpRows + r] * sharedB[r * colsPerThread + c];
            }
        }

        __syncthreads();
    }

    if (col < n) {
        for (int c = 0; c < colsPerThread; ++c) {
            if (col + c < n) {
                atomicAdd(c + c, sum[c]);
            }
        }
    }
}

torch::Tensor matmul_tall_cuda(torch::Tensor A, torch::Tensor B) {
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    auto C = torch::zeros({m, n}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((n + 31) / 32, (m + 7) / 8);

    size_t smemSize = threads.x * threads.y * (warpRows * sizeof(float) + colsPerThread * sizeof(float));

    matmul_kernel<float><<<blocks, threads, smemSize, at::cuda::current_stream()>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), m, k, n);

    return C;
}
"""

matmul_tall_cpp_source = (
    "torch::Tensor matmul_tall_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_tall = load_inline(
    name="matmul_tall",
    cpp_sources=matmul_tall_cpp_source,
    cuda_sources=matmul_tall_source,
    functions=["matmul_tall_cuda"],
    verbose=True,
    extra_cflags=["-gencode=arch=compute_80,code=sm_80", "-Xptxas=-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tall = matmul_tall

    def forward(self, A, B):
        return self.matmul_tall.matmul_tall_cuda(A.cuda(), B.cuda())