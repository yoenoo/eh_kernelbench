import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scan_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void inclusive_scan_kernel(const scalar_t* input, scalar_t* output, int dim_size, int total_elements) {
    extern __shared__ scalar_t shared[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Each block processes a slice along the dim
    int slice_offset = bid * dim_size;
    int block_start = slice_offset + tid;

    // Load data into shared memory
    if (block_start < total_elements) {
        shared[tid] = input[block_start];
    } else {
        shared[tid] = 0;
    }

    __syncthreads();

    // Up-sweep phase
    for (int offset = 1; offset < dim_size; offset <<= 1) {
        int index = 2 * offset - 1;
        if (tid >= offset && tid < dim_size) {
            shared[tid] += shared[tid - offset];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int offset = dim_size >> 1; offset > 0; offset >>= 1) {
        int index = 2 * offset - 1;
        if (tid >= offset) {
            scalar_t tmp = shared[tid - offset];
            __syncthreads(); // Ensure that upper part is computed before read
            if (tid >= offset) {
                shared[tid] += tmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared[0] = 0; // For exclusive scan, we might need to set this, but let's check
    }

    __syncthreads();

    // Now, write the results back
    if (block_start < total_elements) {
        output[block_start] = shared[tid];
    }
}

torch::Tensor inclusive_scan_cuda(torch::Tensor input, int64_t dim) {
    auto input_size = input.sizes().vec();
    int dim_size = input.size(dim);
    auto total_elements = input.numel();

    auto output = torch::empty_like(input);

    dim3 threads(dim_size);
    dim3 blocks(input_size[0]); // Assuming dim is 1, the batch size is first dimension

    int shared_mem = dim_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inclusive_scan_cuda", ([&] {
        inclusive_scan_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            total_elements
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

scan_cpp_source = "torch::Tensor inclusive_scan_cuda(torch::Tensor input, int64_t dim);"

scan_ext = load_inline(
    name="scan_extension",
    cpp_sources=[scan_cpp_source],
    cuda_sources=[scan_source],
    functions=["inclusive_scan_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scan = scan_ext

    def forward(self, x):
        return self.scan.inclusive_scan_cuda(x, self.dim)