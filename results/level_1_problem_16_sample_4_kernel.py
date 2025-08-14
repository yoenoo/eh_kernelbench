import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

template <typename T>
__global__ void custom_matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* C, int M, int K, int N) {
    const int warp_size = 32;
    const int block_size = 256;
    const int num_warps = block_size / warp_size;

    __shared__ T shared_A[block_size * 4];
    __shared__ T shared_B[block_size * 4];

    int tx = threadIdx.x;
    int warp_id = tx / warp_size;
    int lane_id = tx % warp_size;

    int row = blockIdx.z * blockDim.x + tx;
    int col = blockIdx.y * blockDim.x + tx;
    int batch = blockIdx.x;

    T value = 0;

    for (int k = 0; k < (K + blockDim.x - 1) / blockDim.x * blockDim.x; k += blockDim.x) {
        if (k + lane_id < K) {
            shared_A[tx] = A[row * K + k + lane_id];
            shared_B[tx] = B[(k + lane_id) * N + col];
        } else {
            shared_A[tx] = 0;
            shared_B[tx] = 0;
        }
        __syncwarp();

        for (int s = 0; s < warp_size; s++) {
            value += shared_A[tx + s] * shared_B[tx + s];
        }
        __syncwarp();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::zeros({M, N}, options);

    dim3 threads(256);
    dim3 blocks(1, (N + threads.x - 1)/threads.x, (M + threads.x - 1)/threads.x);

    custom_matmul_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );

    return C;
}
"""

matmul_kernel_cpp = "torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B);"

matmul_op = load_inline(
    name="custom_matmul_op",
    cpp_sources=[matmul_kernel_cpp],
    cuda_sources=[matmul_kernel_source],
    functions=["custom_matmul"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.t()
        return self.matmul.custom_matmul(A, B)