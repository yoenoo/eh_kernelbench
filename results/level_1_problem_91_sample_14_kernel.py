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
        __global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, 
                                             int dim_size, int total_size, int dim, int num_dims) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= total_size) return;

            // Calculate the position along the specified dimension
            int* strides = new int[num_dims];
            int idx = index;
            int current_size = 1;
            for (int d = num_dims - 2; d >= 0; --d) {
                strides[d] = current_size;
                current_size *= input.size(d + 1);
            }
            strides[num_dims - 1] = 1;

            int* coords = new int[num_dims];
            for (int d = 0; d < num_dims; d++) {
                coords[d] = idx / strides[d];
                idx = idx % strides[d];
            }

            // Compute reverse cumulative sum along 'dim'
            int start = coords[dim];
            int end = -1;
            int step = -1;
            for (int i = start; i != end; i += step) {
                coords[dim] = i;
                int pos = 0;
                for (int d = 0; d < num_dims; d++) {
                    pos += coords[d] * strides[d];
                }
                output[index] += (i == start) ? input[pos] : output[pos];
            }

            delete[] strides;
            delete[] coords;
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
            const auto dim_size = input.size(dim);
            const auto num_dims = input.dim();
            const auto total_size = input.numel();
            auto output = torch::empty_like(input);

            const int block_size = 256;
            const int grid_size = (total_size + block_size - 1) / block_size;

            // Determine the data type for the kernel
            AT_DISPATCH_ALL_TYPES(input.scalar_type(), "reverse_cumsum_cuda", ([&]{
                reverse_cumsum_kernel<scalar_t><<<grid_size, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, total_size, dim, num_dims);
            }));

            return output;
        }
        """

        # Compile the inline CUDA code for reverse cumulative sum
        reverse_cumsum = load_inline(
            name="reverse_cumsum",
            cpp_sources="",
            cuda_sources=reverse_cumsum_source,
            functions=["reverse_cumsum_cuda"],
            verbose=False
        )

        # Bind the CUDA function to the model
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]