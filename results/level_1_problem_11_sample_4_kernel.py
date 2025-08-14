import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

        # Inline CUDA kernel for 4D tensor-matrix multiplication
        self.tensormatrixmul_cuda = load_inline(
            name='tensormatrixmul',
            cpp_sources=r'''
                torch::Tensor tensormatrixmul_cuda(torch::Tensor A, torch::Tensor B);
            ''',
            cuda_sources=r'''
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <mma.h>

                template <typename scalar_t>
                __global__ void tensormatrixmul_kernel(
                    const scalar_t* __restrict__ A,
                    const scalar_t* __restrict__ B,
                    scalar_t* __restrict__ C,
                    int bdim0, int bdim1, int ldim, int kdim
                ) {
                    int batch_idx = blockIdx.z;
                    int i_idx = blockIdx.y;
                    int j_idx = blockIdx.x;
                    int k = threadIdx.x;

                    scalar_t sum = 0;
                    for (int l = 0; l < ldim; ++l) {
                        sum += A[batch_idx * bdim0*bdim1*ldim + i_idx * bdim1*ldim + j_idx * ldim + l] * 
                               B[l * kdim + k];
                    }
                    C[batch_idx * bdim0*bdim1*kdim + i_idx * bdim1*kdim + j_idx * kdim + k] = sum;
                }

                torch::Tensor tensormatrixmul_cuda(torch::Tensor A, torch::Tensor B) {
                    const int b = A.size(0);
                    const int i = A.size(1);
                    const int j = A.size(2);
                    const int l = A.size(3);
                    const int k = B.size(1);

                    auto C = torch::zeros({b, i, j, k}, A.options());

                    dim3 threads(256);  // Threads per block
                    dim3 blocks(j, i, b);  // Blocks per grid

                    AT_DISPATCH_FLOATING_TYPES(A.type(), "tensormatrixmul_cuda", ([&] {
                        tensormatrixmul_kernel<scalar_t><<<blocks, threads>>>(
                            A.data<scalar_t>(),
                            B.data<scalar_t>(),
                            C.data<scalar_t>(),
                            i, j, l, k);
                    }));

                    cudaDeviceSynchronize();
                    return C;
                }
            ''',
            functions=['tensormatrixmul_cuda'],
            verbose=True
        )

    def forward(self, A, B):
        return self.tensormatrixmul_cuda.tensormatrixmul_cuda(A, B)