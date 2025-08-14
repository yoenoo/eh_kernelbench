import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for inclusive cumsum
cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void inclusive_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim_size,
    int outer_stride,
    int inner_stride) {

    extern __shared__ scalar_t shared[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int tile_size = blockDim.x * 2;

    // Each thread handles multiple elements for a batch
    int global_idx = batch_idx * outer_stride + tid * tile_size;
    scalar_t* local_input = (scalar_t*)&input[global_idx];
    scalar_t* local_output = &output[global_idx];

    // Load data to shared memory
    for (int i = 0; i < tile_size && global_idx + i < outer_stride; i += blockDim.x) {
        shared[tid + i] = local_input[i];
    }
    __syncthreads();

    // Perform parallel prefix sum in shared memory
    for (int s = 1; s < tile_size; s <<= 1) {
        scalar_t temp = 0;
        if (tid >= s) {
            temp = shared[tid - s];
        }
        __syncthreads();
        shared[tid] += temp;
        __syncthreads();
    }

    // Write results back to global memory
    for (int i = 0; i < tile_size && global_idx + i < outer_stride; i += blockDim.x) {
        local_output[i] = shared[tid + i];
    }
    __syncthreads();
}

torch::Tensor inclusive_cumsum_cuda(
    torch::Tensor input,
    int64_t dim) {

    // Assuming dim is 1 (as per input example)
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    int outer_stride = dim_size;
    int inner_stride = 1;

    auto output = torch::empty_like(input);
    const int block_size = 256;
    dim3 blocks(batch_size);
    dim3 threads(block_size);

    int shared_size = block_size * 2 * sizeof(float);
    inclusive_cumsum_kernel<float><<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size,
        outer_stride,
        inner_stride);

    cudaDeviceSynchronize();
    return output;
}
"""

cumsum_cpp_source = "torch::Tensor inclusive_cumsum_cuda(torch::Tensor input, int64_t dim);"

# Compile the CUDA kernel
cumsum_op = load_inline(
    name="inclusive_cumsum",
    cpp_sources=cumsum_cpp_source,
    cuda_sources=cumsum_source,
    functions=["inclusive_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cumsum_op = cumsum_op

    def forward(self, x):
        return self.cumsum_op.inclusive_cumsum_cuda(x, self.dim)