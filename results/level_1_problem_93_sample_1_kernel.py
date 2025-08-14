import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

template <typename scalar_t>
__global__ void masked_cumsum_kernel(const scalar_t* x, const unsigned char* mask,
                                    scalar_t* out, int batch_size, int length,
                                    int dim) {
    // Assuming dim=1, process each sample in the batch independently
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Access the current sample
    int offset = batch_idx * length;

    // Each thread handles one element
    __shared__ scalar_t shared_data[512];
    __shared__ scalar_t partial_sums[512];

    // Load into shared memory only if mask is true
    if (mask[offset + tid] && tid < length) {
        shared_data[tid] = x[offset + tid];
    } else {
        shared_data[tid] = 0.0;
    }
    __syncthreads();

    // Perform parallel prefix sum on shared_data using Blelloch's scan algorithm
    for (int s=1; s <= length; s *= 2) {
        if (tid >= s) {
            if (mask[offset + tid] && mask[offset + tid - s]) {
                shared_data[tid] += shared_data[tid - s];
            }
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (tid < length) {
        out[offset + tid] = shared_data[tid];
    }
}

std::vector<torch::Tensor> masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask) {
    auto dim = x.dim() - 1; // Assuming dim is the last dimension
    auto batch_size = x.size(0);
    auto length = x.size(1);

    torch::Tensor out = torch::zeros_like(x);

    // Set up launch parameters
    dim3 threads_per_block(length);
    dim3 num_blocks(batch_size);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            mask.data_ptr<unsigned char>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            length,
            dim);
    }));

    return {out};
}
"""

masked_cumsum_header = """
std::vector<torch::Tensor> masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask);
"""

masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_header,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        # Ensure mask is of the correct type (byte)
        mask_byte = mask.to(torch.uint8)
        out = self.masked_cumsum.masked_cumsum_cuda(x, mask_byte)[0]
        return out