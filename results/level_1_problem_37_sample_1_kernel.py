import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm calculation and normalization
frobenius_norm_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename scalar_t>
__global__ void compute_norm_kernel(const scalar_t* x, float* norm, int size) {
    extern __shared__ inline __align__(16) float shared_data[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t val = (gid < size) ? static_cast<float>(x[gid]) : 0;
    scalar_t sum = 0;
    sum += val * val;
    // Warp-level reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(norm, sum);
    }
}

template <typename scalar_t>
__global__ void normalize_kernel(const scalar_t* x, scalar_t* y, float norm, int size) {
    norm = sqrt(norm);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = static_cast<scalar_t>(x[idx] / norm);
    }
}

torch::Tensor frobenius_norm_and_normalize_cuda(torch::Tensor x) {
    auto x_contig = x.contiguous();
    auto size = x_contig.numel();
    auto dtype = x_contig.scalar_type();
    auto y = torch::empty_like(x_contig);

    const int block_size = 512;
    const int num_blocks = (size + block_size - 1) / block_size;

    float* norm_dev;
    cudaMalloc(&norm_dev, sizeof(float));
    cudaMemsetAsync(norm_dev, 0, sizeof(float));

    if (dtype == torch::kFloat32) {
        compute_norm_kernel<float><<<num_blocks, block_size, 0, cuda::current_stream()>>>(
            x_contig.data_ptr<float>(), norm_dev, size);
    } else if (dtype == torch::kHalf) {
        compute_norm_kernel<__half><<<num_blocks, block_size, 0, cuda::current_stream()>>>(
            reinterpret_cast<const __half*>(x_contig.data_ptr()), norm_dev, size);
    } else {
        cudaFree(norm_dev);
        TORCH_CHECK(false, "Unsupported data type");
    }

    float norm_host;
    cudaMemcpy(&norm_host, norm_dev, sizeof(float), cudaMemcpyDeviceToHost);

    normalize_kernel<<<num_blocks, block_size, 0, cuda::current_stream()>>>(
        x_contig.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), norm_host, size);

    cudaFree(norm_dev);
    return y;
}
"""

frobenius_norm_norm_cpp_source = """
torch::Tensor frobenius_norm_and_normalize_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
frobenius_ext = load_inline(
    name="frobenius_ext",
    cpp_sources=frobenius_norm_norm_cpp_source,
    cuda_sources=frobenius_norm_norm_source,
    functions=["frobenius_norm_and_normalize_cuda"],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=['-lineinfo', '-O3', '--use_fast_math'],
    extra_cflags=['-O3'],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.norm_and_normalize = frobenius_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_and_normalize.frobenius_norm_and_normalize_cuda(x)