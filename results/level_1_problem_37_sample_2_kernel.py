import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Frobenius norm computation and normalization
frobenius_norm_normalize_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template<typename T>
__global__ void frobenius_norm_normalize(
    const T* __restrict__ x_data,
    T* __restrict__ y_data,
    const int64_t total_elements,
    const int64_t batch_size,
    const int64_t features,
    const int64_t dim1,
    const int64_t dim2
) {
    using BlockReduce = cub::BlockReduce<T, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T sum = static_cast<T>(0);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        sum += x_data[idx] * x_data[idx];
    }

    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    T block_sum;
    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<T*>(&block_sum), sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sum = block_sum;
    }
    __syncthreads();

    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());

    if (threadIdx.x == 0) {
        T norm = sqrt(sum);
        if (norm == 0) norm = 1; // avoid division by zero
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
            y_data[idx] = x_data[idx] / norm;
        }
    }
}

std::vector<torch::Tensor> frobenius_norm_normalize_cuda(torch::Tensor x) {
    auto total_elements = x.numel();
    auto y = torch::empty_like(x);
    auto block_size = 256;
    auto grid_size = (total_elements + block_size - 1) / block_size;

    dim3 grid(grid_size);
    dim3 block(block_size);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "frobenius_norm_normalize_cuda", ([&] {
        frobenius_norm_normalize<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            total_elements,
            x.size(0),
            x.size(1),
            x.size(2),
            x.size(3)
        );
    }));

    cudaDeviceSynchronize();
    return {y};
}
"""

frobenius_norm_normalize_cpp_source = (
    "std::vector<torch::Tensor> frobenius_norm_normalize_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Frobenius norm normalization
frobenius_norm_normalize = load_inline(
    name="frobenius_norm_normalize",
    cpp_sources=frobenius_norm_normalize_cpp_source,
    cuda_sources=frobenius_norm_normalize_source,
    functions=["frobenius_norm_normalize_cuda"],
    verbose=True,
    extra_cflags=["-DCUB_VERSION_MIN_1_12_0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_norm_normalize = frobenius_norm_normalize

    def forward(self, x):
        return self.frobenius_norm_normalize.frobenius_norm_normalize_cuda(x)[0]

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []