import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_normalize = load_inline(
            name='l1_normalize',
            cpp Sources='',
            cuda Sources='''
                #include <torch/extension.h>
                #include <cuda_runtime.h>

                template <typename scalar_t>
                __global__ void l1_normalize_kernel(
                    const scalar_t* __restrict__ x,
                    scalar_t* __restrict__ output,
                    const size_t batch_size,
                    const size_t dim) {
                    int batch_idx = blockIdx.x;
                    int element_idx = threadIdx.x;

                    __shared__ scalar_t sum[256];
                    sum[threadIdx.x] = 0.0;

                    for (int i = element_idx; i < dim; i += blockDim.x) {
                        sum[threadIdx.x] += fabs(x[batch_idx * dim + i]);
                    }

                    __shared__ scalar_t mean;
                    if (threadIdx.x == 0) {
                        scalar_t total = 0.0;
                        for (int i = 0; i < blockDim.x; ++i) {
                            total += sum[i];
                        }
                        mean = total / dim;
                    }
                    __syncthreads();

                    if (element_idx < dim) {
                        output[batch_idx * dim + element_idx] = x[batch_idx * dim + element_idx] / mean;
                    }
                }

                torch::Tensor l1_normalize_cuda(torch::Tensor x) {
                    const int batch_size = x.size(0);
                    const int dim = x.size(1);

                    auto output = torch::empty_like(x);

                    const dim3 block(min(dim, 256));
                    const dim3 grid(batch_size);

                    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "l1_normalize_cuda", ([&] {
                        l1_normalize_kernel<scalar_t><<<grid, block>>>(
                            x.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size,
                            dim);
                    }));

                    return output;
                }
            ''',
            functions=['l1_normalize_cuda'],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_normalize.l1_normalize_cuda(x)