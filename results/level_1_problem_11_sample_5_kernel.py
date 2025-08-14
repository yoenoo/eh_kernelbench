import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the custom CUDA kernel for the einsum operation
        self.einsum_kernel = load_inline(
            name="einsum_bijl_lk_bijk",
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __global__ void einsum_bijl_lk_bijk_kernel(
                const scalar_t* __restrict__ A,
                const scalar_t* __restrict__ B,
                scalar_t* __restrict__ C,
                const int b,
                const int i,
                const int j,
                const int l,
                const int k
            ) {{
                const int batch_idx = blockIdx.z;
                const int i_idx = blockIdx.y;
                const int j_idx = blockIdx.x;
                const int k_idx = threadIdx.x;

                scalar_t sum = 0;
                for (int l_idx = 0; l_idx < l; ++l_idx) {{
                    sum += A[batch_idx * i * j * l + i_idx * j * l + j_idx * l + l_idx] *
                           B[l_idx * k + k_idx];
                }}
                C[batch_idx * i * j * k + i_idx * j * k + j_idx * k + k_idx] = sum;
            }}

            torch::Tensor einsum_bijl_lk_bijk_cuda(torch::Tensor A, torch::Tensor B) {{
                const int b = A.size(0);
                const int i = A.size(1);
                const int j = A.size(2);
                const int l = A.size(3);
                const int k = B.size(1);

                auto C = torch::empty({{b, i, j, k}}, A.options());

                const dim3 block_dim(k);
                const dim3 grid_dim(j, i, b);

                AT_DISPATCH_FLOATING_TYPES(A.type(), "einsum_bijl_lk_bijk", ([&] {{
                    einsum_bijl_lk_bijk_kernel<scalar_t><<<grid_dim, block_dim>>>(
                        A.data<scalar_t>(),
                        B.data<scalar_t>(),
                        C.data<scalar_t>(),
                        b, i, j, l, k
                    );
                }}));

                return C;
            }}
            """,
            functions=["einsum_bijl_lk_bijk_cuda"],
            verbose=True
        )

    def forward(self, A, B):
        return self.einsum_kernel.einsum_bijl_lk_bijk_cuda(A, B)

def get_inputs():
    # Maintaining original input generation for compatibility
    A = torch.rand(b, i, j, l).cuda()
    B = torch.rand(l, k).cuda()
    return [A, B]

def get_init_inputs():
    return []