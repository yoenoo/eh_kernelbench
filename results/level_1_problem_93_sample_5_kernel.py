import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        masked_cumsum_kernel = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void masked_cumsum_kernel(const scalar_t* x, const unsigned char* mask, scalar_t* out, int dim_size, int batch_size, int total_size) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= total_size) return;

            int batch = index / dim_size;
            int pos = index % dim_size;

            scalar_t sum = 0;
            for (int i = 0; i <= pos; ++i) {
                int flat_idx = batch * dim_size + i;
                if (mask[flat_idx]) {
                    sum += x[flat_idx];
                }
            }
            out[index] = sum;
        }

        torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
            int64_t dim_size = x.size(dim);
            auto out = torch::empty_like(x);

            int block_size = 256;
            int grid_size = (x.numel() + block_size - 1) / block_size;

            auto stream = at::cuda::getCurrentCUDAStream();
            AT_DISPATCH_ALL_TYPES_AND2(x.scalar_type(), "masked_cumsum_cuda", ([&] {
                masked_cumsum_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    mask.data_ptr<unsigned char>(),
                    out.data_ptr<scalar_t>(),
                    dim_size,
                    x.size(0),
                    x.numel()
                );
            }));

            return out;
        }
        """

        self.masked_cumsum = load_inline(
            name="masked_cumsum",
            cpp_sources="",
            cuda_sources=masked_cumsum_kernel,
            functions=["masked_cumsum_cuda"],
            verbose=True,
            with_cuda=True
        )

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)