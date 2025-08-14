import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

        # Define and load the custom CUDA kernel for reverse cumulative sum
        reverse_cumsum_cuda_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int64_t size, int64_t dim_size, int64_t before_dim, int64_t after_dim, int64_t dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            // Compute indices
            int64_t d = idx / (before_dim * dim_size) * dim_size;
            int64_t pos = (idx % (before_dim * dim_size)) / before_dim;
            int64_t rest = idx % before_dim;

            // Reverse the dimension
            pos = (dim_size - 1) - pos;

            // Compute original and output indices
            int64_t in_idx = d + pos * before_dim + rest;
            int64_t out_idx = idx;

            // Compute cumulative sum
            scalar_t sum = 0;
            for (int i = 0; i <= pos; i++) {
                int64_t current_in = d + (dim_size - 1 - i) * before_dim + rest;
                sum += input[current_in];
            }
            output[out_idx] = sum;
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim) {
            auto output = torch::empty_like(input);
            auto input_data = input.contiguous();

            int64_t ndim = input.dim();
            auto sizes = input.sizes().vec();
            int64_t dim_size = sizes[dim];
            int64_t before_dim = 1;
            for (int i = 0; i < dim; i++)
                before_dim *= sizes[i];
            int64_t after_dim = 1;
            for (int i = dim + 1; i < ndim; i++)
                after_dim *= sizes[i];
            int64_t total_elements = input.numel();

            const int block_size = 256;
            const int num_blocks = (total_elements + block_size - 1) / block_size;

            // Dispatch kernel based on the data type
            AT_DISPATCH_FLOATING_TYPES(input.type(), "reverse_cumsum_cuda", ([&] {
                reverse_cumsum_kernel<scalar_t><<<num_blocks, block_size>>>(
                    input_data.data<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    total_elements,
                    dim_size,
                    before_dim,
                    after_dim,
                    dim
                );
            }));

            cudaDeviceSynchronize();
            return output;
        }
        """

        reverse_cumsum_cpp_source = (
            "torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim);"
        )

        self.reverse_cumsum_cuda = load_inline(
            name="reverse_cumsum_cuda",
            cpp_sources=reverse_cumsum_cpp_source,
            cuda_sources=reverse_cumsum_cuda_source,
            functions=["reverse_cumsum_cuda"],
            verbose=True,
            with_cuda=True,
        )

    def forward(self, x):
        return self.reverse_cumsum_cuda.reverse_cumsum_cuda(x, self.dim)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]