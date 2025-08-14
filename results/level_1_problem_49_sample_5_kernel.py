import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max reduction
max_reduction_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    int dim_size,
    int outer_dim,
    int inner_dim
) {
    extern __shared__ scalar_t shared[];
    int tid = threadIdx.x;
    int outer_id = blockIdx.x;
    scalar_t max_val = -FLT_MAX;

    // Compute the starting index for this block
    int index = outer_id * dim_size * inner_dim + tid;

    // Each thread reads an element and computes partial max
    for (int i = 0; i < inner_dim; i++) {
        int pos = index + i * dim_size;
        if (pos < outer_id * dim_size * inner_dim + dim_size * inner_dim) {
            scalar_t val = input[pos];
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    // Write partial max to shared memory
    shared[tid] = max_val;

    // Synchronize to ensure all data is in shared memory
    __syncthreads();

    // Perform block-wide reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }

    // Write the final result to output
    if (tid == 0) {
        output[outer_id] = shared[0];
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(dim);
    const int other_dims_product = input.numel() / (batch_size * dim_size);

    const int threads_per_block = 256;
    const dim3 blocks(batch_size * other_dims_product);
    const dim3 threads(threads_per_block);

    // Calculate shared memory size
    size_t shared_mem_size = threads_per_block * sizeof(float);

    auto output = torch::empty({batch_size, other_dims_product}, input.options());

    // Launch kernel with necessary dimensions and shared memory
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_reduction_cuda", ([&] {
        max_reduction_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            batch_size,
            other_dims_product
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

max_reduction_cpp_source = (
    "torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);"
)

# Compile the inline CUDA code
max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()  # Ensure tensor is on GPU
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension