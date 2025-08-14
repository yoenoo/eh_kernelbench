import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l2_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y,
                              int batch_size, int dim) {
    int sample_idx = blockIdx.x;

    __shared__ float shared_sums[32]; // Assuming block size divides 32

    scalar_t sum = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        scalar_t val = x[sample_idx * dim + i];
        sum += val * val;
    }

    // Block reduction using shared memory
    shared_sums[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float norm = sqrtf(shared_sums[0]);
        // Add small epsilon for numerical stability
        if (norm < 1e-12f) {
            norm = 1e-12f;
        }
        for (int i = 0; i < dim; ++i) {
            int idx = sample_idx * dim + i;
            y[idx] = x[idx] / norm;
        }
    }
    __syncthreads();
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    auto y = torch::empty_like(x);

    dim3 blocks(batch_size);
    dim3 threads(std::min(256, dim));
    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(), "l2_norm_cuda", ([&]{
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));

    cudaDeviceSynchronize();
    return y;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor x);"
)

l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm.l2_norm_cuda(x)