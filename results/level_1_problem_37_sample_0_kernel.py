import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename scalar_t>
__global__ void frobenius_norm_kernel(const scalar_t* __restrict__ x_data, float* norm, int64_t size) {
    extern __shared__ scalar_t shared[];
    int tid = threadIdx.x;
    scalar_t sum = 0.0;

    for (int i = blockIdx.x * blockDim.x + tid; i < size; i += blockDim.x * gridDim.x) {
        scalar_t val = x_data[i];
        sum += val * val;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm, static_cast<float>(shared[0]));
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    auto x_contig = x.contiguous();
    int64_t size = x_contig.numel();
    const int block_size = 256;
    const int grid_size = std::min(32768, (size + block_size - 1) / block_size);

    float* norm_ptr;
    cudaMallocManaged(&norm_ptr, sizeof(float));
    *norm_ptr = 0.0f;

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(grid_size);
    dim3 threads(block_size);
    size_t shared_size = sizeof(float) * block_size;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "frobenius_norm_cuda", ([&] {
        frobenius_norm_kernel<scalar_t><<<blocks, threads, shared_size, stream>>>(
            x_contig.data_ptr<scalar_t>(), norm_ptr, size);
    }));

    cudaStreamSynchronize(stream);
    cudaFree(norm_ptr);
    return torch::sqrt(torch::tensor(*norm_ptr, device=x.device()));
}

torch::Tensor frobenius_normalize_cuda(torch::Tensor x) {
    auto norm = frobenius_norm_cuda(x);
    auto inv_norm = 1.0 / norm;
    auto y = x * inv_norm;
    return y;
}
"""

frobenius_normalize_cpp_source = """
torch::Tensor frobenius_normalize_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
frobenius_normalize = load_inline(
    name="frobenius_normalize",
    cpp_sources=frobenius_normalize_cpp_source,
    cuda_sources=frobenius_source,
    functions=["frobenius_normalize_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_normalize = frobenius_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_normalize.frobenius_normalize_cuda(x)