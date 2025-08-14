import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the custom CUDA kernel for batched matrix multiplication
        matmul_3d_2d_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void batched_matmul_kernel(const scalar_t* __restrict__ a,
                                             const scalar_t* __restrict__ b,
                                             scalar_t* __restrict__ c,
                                             int N, int M, int K, int L) {
            int n = blockIdx.x;
            int m = blockIdx.y;
            int l = threadIdx.x;

            scalar_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += a[n * M * K + m * K + k] * b[k * L + l];
            }
            c[n * M * L + m * L + l] = sum;
        }

        torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b) {
            const int N = a.size(0);
            const int M = a.size(1);
            const int K = a.size(2);
            const int L = b.size(1);

            auto c = torch::empty({N, M, L}, a.options());

            dim3 threads_per_block(std::min(L, 512));
            dim3 num_blocks(M, N);

            AT_DISPATCH_ALL_TYPES(a.scalar_type(), "batched_matmul_cuda", ([&] {
                batched_matmul_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                    a.data<scalar_t>(),
                    b.data<scalar_t>(),
                    c.data<scalar_t>(),
                    N, M, K, L);
            }));

            return c;
        }
        """

        matmul_3d_2d_cpp_source = "torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b);"

        # Compile the inline CUDA code
        self.matmul_3d_2d = load_inline(
            name="batched_matmul",
            cpp_sources=matmul_3d_2d_cpp_source,
            cuda_sources=matmul_3d_2d_source,
            functions=["batched_matmul_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, A, B):
        return self.matmul_3d_2d.batched_matmul_cuda(A, B)

# Note: get_inputs and get_init_inputs remain unchanged as per original code