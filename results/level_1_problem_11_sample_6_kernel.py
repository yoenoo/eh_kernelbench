import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define and load custom CUDA kernel for the einsum operation
        einsum_kernel_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        __global__ void einsum_bijl_lk_to_bijk(const float* A, const float* B, float* C,
                                              int b, int i, int j, int l, int k) {
            int batch_idx = blockIdx.z;
            int output_i = blockIdx.y;
            int output_j = blockIdx.x;
            int tid = threadIdx.x;

            __shared__ float shared_A[32];
            __shared__ float shared_B[32];

            float sum = 0.0;

            for (int block = 0; block < (l + blockDim.x - 1) / blockDim.x; ++block) {
                int l_idx = block * blockDim.x + tid;
                if (l_idx < l) {
                    shared_A[tid] = A[batch_idx * i * j * l + output_i * j * l + output_j * l + l_idx];
                    shared_B[tid] = B[l_idx * k + tid];
                } else {
                    shared_A[tid] = 0.0;
                    shared_B[tid] = 0.0;
                }
                __syncthreads();

                for (int s = 0; s < blockDim.x; ++s) {
                    sum += shared_A[s] * shared_B[s];
                }
                __syncthreads();
            }

            int c_idx = batch_idx * i * j * k + output_i * j * k + output_j * k + tid;
            C[c_idx] = sum;
        }

        torch::Tensor custom_einsum_cuda(torch::Tensor A, torch::Tensor B) {
            const int b = A.size(0);
            const int i = A.size(1);
            const int j = A.size(2);
            const int l = A.size(3);
            const int k = B.size(1);

            auto C = torch::empty({b, i, j, k}, A.options());

            dim3 threads(256);  // Threads per block
            dim3 blocks(j, i, b);  // Blocks per grid (each block computes one output (i,j,k) slice)

            einsum_bijl_lk_to_bijk<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                       C.data_ptr<float>(), b, i, j, l, k);

            return C;
        }
        """

        einsum_kernel_header = """
        torch::Tensor custom_einsum_cuda(torch::Tensor A, torch::Tensor B);
        """

        self.einsum_op = load_inline(
            name='einsum_op',
            cpp_sources=einsum_header,
            cuda_sources=einsum_kernel_source,
            functions=['custom_einsum_cuda'],
            verbose=True
        )

    def forward(self, A, B):
        return self.einsum_op.custom_einsum_cuda(A, B)