import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

        # Define and compile the custom CUDA kernel for sum reduction
        reduction_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fast_reduction_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     const int batch_size,
                                     const int reduce_dim_size,
                                     const int remaining_dims,
                                     const int dim) {{
    int batch_idx = blockIdx.x;
    int remaining_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (remaining_idx >= remaining_dims) return;
    
    int output_idx = batch_idx * remaining_dims + remaining_idx;
    atomicExch(output + output_idx, 0);
    
    for (int i = threadIdx.x; i < reduce_dim_size; i += blockDim.x) {{
        int input_idx = batch_idx * reduce_dim_size * remaining_dims
                     + i * remaining_dims
                     + remaining_idx;
        atomicAdd(output + output_idx, input[input_idx]);
    }}
}}

std::vector<int64_t> get_output_size(torch::Tensor input, int dim) {{
    auto input_shape = input.sizes().vec();
    input_shape[dim] = 1;
    return input_shape;
}}

torch::Tensor fast_reduction_cuda(torch::Tensor input, int dim) {{
    const int batch_size = input.size(0);
    const int reduce_dim_size = input.size(dim);
    int remaining_dims = 1;
    for (int i = 1; i < input.dim(); ++i) {{
        if (i != dim) remaining_dims *= input.size(i);
    }}
    
    auto output = torch::empty(get_output_size(input, dim), 
                              input.options());
    
    const int blocks_per_batch = (remaining_dims + 31) / 32;
    dim3 threads(256, 32); // 256 threads per block (32x8)
    dim3 grid(batch_size, blocks_per_batch);
    
    fast_reduction_kernel<float><<<grid, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        reduce_dim_size,
        remaining_dims,
        dim
    );
    
    return output;
}}
"""

        reduction_cpp = """
std::vector<int64_t> get_output_size(torch::Tensor input, int dim);
torch::Tensor fast_reduction_cuda(torch::Tensor input, int dim);
"""

        # Compile with CUDA
        self.fast_reduction = load_inline(
            name="fast_reduction",
            cpp_sources=reduction_cpp,
            cuda_sources=reduction_source,
            functions=["fast_reduction_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fast_reduction.fast_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [reduce_dim]