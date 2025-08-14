import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template<typename T>
__global__ void compute_frobenius_norm(const T* data, T* norm, int64_t size) {
    extern __shared__ char shared_storage[];
    T* shared = reinterpret_cast<T*>(shared_storage);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T val = (idx < size) ? data[idx] * data[idx] : 0;
    cub::BlockReduce<T, 256> reduce(cub::BLOCK_DIM);
    T block_sum = reduce.Sum(val);
    if (threadIdx.x == 0) {
        *norm += block_sum;
    }
    __syncthreads();
}

template<typename T>
void compute_norm_cuda(torch::Tensor data, torch::Tensor norm) {
    auto size = data.numel();
    auto num_threads = 1024;
    auto num_blocks = (size + num_threads - 1) / num_threads;
    auto smem_size = sizeof(T) * 256;
    compute_frobenius_norm<T><<<num_blocks, num_threads, smem_size>>>(
        data.data_ptr<T>(), norm.data_ptr<T>(), size);
}

torch::Tensor frobenius_normalize_cuda(torch::Tensor x) {
    auto norm = torch::zeros(1, x.dtype(), device=x.device());
    if (x.dtype() == torch::kFloat32) {
        compute_norm_cuda<float>(x, norm);
        norm = torch::sqrt(norm);
    } else if (x.dtype() == torch::kFloat64) {
        compute_norm_cuda<double>(x, norm);
        norm = torch::sqrt(norm);
    }
    return x / norm.item();
}

"""

frobenius_norm_header = "torch::Tensor frobenius_normalize_cuda(torch::Tensor x);"

frobenius_normalize = load_inline(
    name="frobenius_normalize",
    cpp_sources=frobenius_norm_header,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_normalize_cuda"],
    verbose=True,
    extra_cflags=["-gencode=arch=compute_70,code=sm_70", "-I/usr/local/cuda/include"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.frobenius_normalize = frobenius_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_normalize.frobenius_normalize_cuda(x)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []