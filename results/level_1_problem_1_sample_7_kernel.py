import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cuda_matmul = load_inline(
            name='cuda_matmul',
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                
                __global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {{
                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    if (row < N && col < N) {{
                        float sum = 0.0;
                        for (int k = 0; k < N; ++k) {{
                            sum += A[row * N + k] * B[k * N + col];
                        }}
                        C[row * N + col] = sum;
                    }}
                }}
                
                torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {{
                    int N = A.size(0);
                    const int block_size = 32;
                    dim3 block(block_size, block_size);
                    dim3 grid((N + block_size - 1)/block_size, (N + block_size - 1)/block_size);
                    
                    auto C = torch::empty({{N, N}}, A.options());
                    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
                    
                    return C;
                }}
            """,
            functions=['matmul_cuda'],
            verbose=True
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.cuda_matmul.matmul_cuda(A, B)

def get_inputs():
    N = 2048 * 2
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []