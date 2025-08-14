import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

        # Define and load the custom CUDA kernel
        reverse_cumsum_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(const scalar_t* x, scalar_t* out, int64_t size, int64_t dim_size, int64_t dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            int64_t dim_offset = idx / dim_size;
            int64_t pos_in_dim = idx % dim_size;
            int64_t rev_pos = (dim_size - 1 - pos_in_dim);

            // Compute the reverse cumulative sum
            scalar_t sum = 0;
            for (int i = 0; i <= pos_in_dim; ++i) {
                int64_t rev_idx = dim_offset * dim_size + (dim_size - 1 - i);
                sum += x[rev_idx];
            }
            out[idx] = sum;
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor x, int64_t dim) {
            const auto dims = x.sizes().vec();
            int64_t dim_size = dims[dim];
            int64_t size = x.numel();

            auto out = torch::empty_like(x);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            // Launch kernel with template for float or double
            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda", ([&] {
                reverse_cumsum_kernel<scalar_t><<<num_blocks, block_size>>>(
                    x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), size, dim_size, dim);
            }));

            return out;
        }
        """

        reverse_cumsum_cpp_source = (
            "torch::Tensor reverse_cumsum_cuda(torch::Tensor x, int64_t dim);"
        )

        # Load the CUDA kernel into Python
        self.reverse_cumsum = load_inline(
            name="reverse_cumsum",
            cpp_sources=[reverse_cumsum_cpp_source],
            cuda_sources=[reverse_cumsum_source],
            functions=["reverse_cumsum_cuda"],
            verbose=True,
            with_cuda=True,
        )

    def forward(self, x):
        # Call the custom CUDA kernel directly
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)