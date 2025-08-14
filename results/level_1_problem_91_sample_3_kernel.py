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
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int dim_size, int total_elements, int step) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements) return;

            int dim_offset = (idx / dim_size) * dim_size;
            int local_idx = idx % dim_size;

            scalar_t sum = 0;
            for (int i = 0; i <= local_idx; i++) {
                sum += input[dim_offset + (dim_size - 1 - i)];
            }
            output[idx] = sum;
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
            auto output = torch::empty_like(input);
            auto dims = input.sizes().vec();
            int total_elements = input.numel();
            int dim_size = dims[dim];

            int block_size = 256;
            int grid_size = (total_elements + block_size - 1) / block_size;

            // Calculate the step for moving along the dimension
            int step = 1;
            for (int i = dim + 1; i < dims.size(); i++) {
                step *= dims[i];
            }

            // Launch kernel for each element
            reverse_cumsum_kernel<float><<<grid_size, block_size>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                dim_size,
                total_elements,
                step
            );

            return output;
        }
        """

        reverse_cumsum_cpp = """
        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);
        """

        # Load the CUDA kernel
        self.reverse_cumsum = load_inline(
            name="reverse_cumsum",
            cpp_sources=reverse_cumsum_cpp,
            cuda_sources=reverse_cumsum_source,
            functions=["reverse_cumsum_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]