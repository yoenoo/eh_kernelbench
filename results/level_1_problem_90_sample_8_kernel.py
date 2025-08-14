import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Define the custom CUDA kernel for cumulative product
        cumulative_product_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cumulative_product_kernel(const scalar_t* input, scalar_t* output, int dim_size, int outer_size, int inner_size, int dim) {
    extern __shared__ scalar_t shared[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = index / inner_size;
    int element_idx = index % inner_size;

    if (index >= dim_size) return;

    // Load input data into shared memory
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        int input_offset = batch_idx * inner_size + i;
        shared[i] = input[input_offset];
    }
    __syncthreads();

    // Perform parallel prefix product
    for (int stride = 1; stride <= element_idx; stride *= 2) {
        int min_idx = element_idx - stride;
        if (min_idx >= 0) {
            shared[threadIdx.x] += shared[min_idx];
        }
        __syncthreads();
    }

    // Store result back
    output[index] = shared[threadIdx.x];
    __syncthreads();
}

torch::Tensor cumulative_product_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int input_dim_size = input.size(dim);
    const int outer_size = input.size(0);
    const int inner_size = 1;
    for (int i = 1; i < input.dim(); ++i) {
        if (i != dim) {
            inner_size *= input.size(i);
        }
    }

    auto output = torch::empty_like(input);
    int block_size = 256;
    int grid_size = (input_dim_size + block_size - 1) / block_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int shared_mem = inner_size * sizeof(float);

    // Launch kernel
    cumulative_product_kernel<float><<<grid_size, block_size, shared_mem, stream>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input_dim_size, outer_size, inner_size, dim);

    return output;
}
"""

        cumulative_product_cpp_source = """
torch::Tensor cumulative_product_cuda(torch::Tensor input, int dim);
"""

        # Compile the inline CUDA code for cumulative product
        self.cumulative_product = load_inline(
            name="cumulative_product",
            cpp_sources=cumulative_product_cpp_source,
            cuda_sources=cumulative_product_source,
            functions=["cumulative_product_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, x):
        return self.cumulative_product.cumulative_product_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]