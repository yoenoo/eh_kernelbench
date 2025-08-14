import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

        # Define the custom CUDA kernel for reverse cumulative sum
        reverse_cumsum_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int dim_size, int total_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_size)
                return;

            int dim_stride = 1;
            for (int d = 0; d < dim; d++)
                dim_stride *= input.size(d);

            int outer = idx / (dim_size * dim_stride);
            int inner_offset = idx % dim_stride;
            int dim_offset = (idx / dim_stride) % dim_size;

            int original_idx = outer * dim_size * dim_stride + (dim_size - 1 - dim_offset) * dim_stride + inner_offset;

            scalar_t sum = 0;
            for (int i = 0; i <= dim_offset; i++) {
                int current_idx = outer * dim_size * dim_stride + (dim_size - 1 - i) * dim_stride + inner_offset;
                sum += input[current_idx];
            }
            output[original_idx] = sum;
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
            auto output = torch::empty_like(input);
            int total_size = output.numel();
            int dim_size = input.size(dim);

            const int block_size = 256;
            const int grid_size = (total_size + block_size - 1) / block_size;

            int dim_ = dim;
            reverse_cumsum_kernel<float><<<grid_size, block_size>>>(
                input.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                output.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                dim_size,
                total_size,
                dim_);

            return output;
        }
        """

        # Compile the inline CUDA code for reverse cumulative sum
        self.reverse_cumsum = load_inline(
            name="reverse_cumsum",
            cpp_sources="",
            cuda_sources=reverse_cumsum_source,
            functions=["reverse_cumsum_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)