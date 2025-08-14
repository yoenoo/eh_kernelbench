import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def triangular_matmul_cuda(A, B):
    batch_size, N, _ = A.shape
    C = torch.empty_like(A)

    triangular_matmul_source = f"""
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    __global__ void triangular_matmul_kernel(float* A, float* B, float* C, int N) {{
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= N || col >= N) return;

        if (row < col) {{
            C[row * N + col] = 0.0f;
            return;
        }}

        float sum = 0.0f;
        for (int k = 0; k < N; k++) {{
            if (k > row || k > col) continue;
            sum += A[row * N + k] * B[k * N + col];
        }}
        C[row * N + col] = sum;
    }}

    torch::Tensor triangular_matmul(torch::Tensor A, torch::Tensor B) {{
        int N = A.size(1);
        dim3 threads(32, 8);
        dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
        auto C = torch::empty_like(A);

        triangular_matmul_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
        );
        cudaDeviceSynchronization();
        return C;
    }}
    """
    
    triangular_matmul_cpp_source = (
        "torch::Tensor triangular_matmul(torch::Tensor A, torch::Tensor B);"
    )

    module = load_inline(
        name="triangular_matmul",
        cpp_sources=triangular_matmul_cpp_source,
        cuda_sources=triangular_matmul_source,
        functions=["triangular_matmul"],
        verbose=True
    )
    return module.triangular_matmul(A, B)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A, B):
        return triangular_matmul_cuda(A, B)

def get_inputs():
    M = 4096
    A = torch.rand(M, M).cuda()
    B = torch.rand(M, M).cuda()
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []