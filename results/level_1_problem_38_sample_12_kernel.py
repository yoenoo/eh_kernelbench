import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void l1_norm_forward_kernel(const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> x,
                                      torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> output,
                                      int batch_size, int dim) {

    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    __shared__ scalar_t sum[32]; // Assuming maximum 32 threads per block for sum reduction

    if (element_idx < dim) {
        scalar_t val = x[batch_idx][element_idx];
        sum[threadIdx.x] = fabs(val);
    } else {
        sum[threadIdx.x] = 0;
    }

    __syncthreads();

    // Reduce the sum across threads in the block
    for (int s = 1; s < gridDim.y; s <<= 1) {
        if (threadIdx.x % (2 * s) == 0) {
            sum[threadIdx.x] += sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (element_idx == 0) {
        scalar_t mean_val = sum[0] / dim;
        for (int i = 0; i < dim; ++i) {
            output[batch_idx][i] = x[batch_idx][i] / mean_val;
        }
    }
}

torch::Tensor l1_norm_forward_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    auto output = torch::empty_like(x);

    dim3 blocks(batch_size);
    dim3 threads(dim); // Each thread handles one element, but needs to have threads >= dim for the shared memory approach. However, this may exceed the maximum thread per block limit.

    // Since the maximum number of threads per block is 1024, need to adjust the block size accordingly
    // Given dim is 65535, this approach may not be feasible. Need to redesign the kernel.

    // Alternative approach with block size divided into multiple threads per batch:
    // Let's have each block handle a single batch sample, and split the dimension into multiple threads.

    const int threads_per_block = 256;
    dim3 grid(batch_size);
    dim3 block(threads_per_block);

    // Using shared memory per block for reduction
    __shared__ float shared_mem[threads_per_block];

    // Re-writing kernel to use proper block-wise reduction:
    l1_norm_forward_kernel<scalar_t><<<grid, block>>>(
        x.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
        output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
        batch_size, dim);

    return output;
}
"""

l1_norm_cpp_source = """
torch::Tensor l1_norm_forward_cuda(torch::Tensor x);
"""

l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_forward_cuda"],
    verbose=True,
    extra_cflags=["-D.Aggressive"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_forward_cuda(x)