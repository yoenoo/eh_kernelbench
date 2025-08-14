import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_normalize_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>

template <typename scalar_t>
__global__ void l1_normalize_kernel(const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> x,
                                   torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> out,
                                   const int batch_size,
                                   const int dim) {
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ scalar_t sum[32]; // Assuming blockDim.x <= 32 for simplicity, but need to adjust based on actual block size
    sum[tid] = 0;

    // Compute partial sums for each thread
    for (int i = tid; i < dim; i += blockDim.x) {
        sum[tid] += fabs(x[batch_id][i]);
    }

    // Sequential reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sum[tid] += sum[tid + s];
        }
    }

    __syncthreads();

    scalar_t total_sum = sum[0];

    // Broadcast and normalize
    if (tid < dim) {
        out[batch_id][tid] = x[batch_id][tid] / total_sum;
    }
}

std::tuple<torch::Tensor, torch::Tensor> l1_normalize_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    auto out = torch::empty_like(x);

    // Define block and grid dimensions
    const int block_size = 256; // Tunable parameter
    dim3 blocks(batch_size);
    dim3 threads(block_size);

    // Assuming half type for potential mixed precision
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "l1_normalize_cuda", ([&] {
        l1_normalize_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            batch_size,
            dim
        );
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(out);
}
"""

cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> l1_normalize_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize_cuda",
    cpp_sources=cpp_source,
    cuda_sources=l1_normalize_source,
    functions=["l1_normalize_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.l1_normalize.l1_normalize_cuda(x)[0]
        return out