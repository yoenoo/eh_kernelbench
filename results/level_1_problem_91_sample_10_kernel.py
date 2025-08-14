import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for reverse cumulative sum
reverse_cumsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int dim_size, int total_elements, int dim_stride, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    int base = index / dim_size;
    int offset = index % dim_size;

    // Compute the original position in the dimension
    int pos = (dim_size - 1 - offset) * dim_stride + base * dim_size + offset;

    extern __shared__ scalar_t shared_data[];
    scalar_t* sdata = shared_data;

    // Load input into shared memory
    sdata[threadIdx.x] = (threadIdx.x < dim_size) ? input[pos - (dim_size - 1 - threadIdx.x)*dim_stride] : 0;

    __syncthreads();

    // Perform parallel prefix sum (reverse cumulative sum)
    for (int stride = 1; stride <= offset + 1; stride *= 2) {
        int index = 2 * stride - 1 - offset;
        if (index >= 0 && index < dim_size) {
            sdata[threadIdx.x] += sdata[index];
        }
        __syncthreads();
    }

    output[pos] = sdata[threadIdx.x];
}

std::vector<torch::Tensor> reverse_cumsum_cuda(torch::Tensor input, int dim) {
    const int64_t* input_size = input.sizes().data();
    int dim_size = input.size(dim);
    int total_elements = input.numel();
    int batch_dims = 1;
    for (int i = 0; i < dim; i++) {
        batch_dims *= input_size[i];
    }
    int dim_stride = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        dim_stride *= input_size[i];
    }

    auto output = torch::empty_like(input);
    int block_size = dim_size;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    reverse_cumsum_kernel<float><<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        dim_size, 
        total_elements, 
        dim_stride, 
        dim
    );

    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reverse_cumsum_cuda", &reverse_cumsum_cuda, "Reverse Cumulative Sum CUDA Kernel");
}
"""

reverse_cumsum_cuda_header = """
#include <torch/extension.h>
std::vector<torch::Tensor> reverse_cumsum_cuda(torch::Tensor input, int dim);
"""

# Compile the custom CUDA kernel
reverse_cumsum = load_inline(
    name='reverse_cumsum_cuda',
    cpp_sources=reverse_cumsum_cuda_header,
    cuda_sources=reverse_cumsum_cuda_source,
    functions=['reverse_cumsum_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use custom CUDA kernel for reverse cumsum
        return reverse_cumsum.reverse_cumsum_cuda(x, self.dim)[0]

def get_inputs():
    # Generating inputs as per original setup (assuming dim=1 and batch_size=32768)
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]  # dim=1 as per original