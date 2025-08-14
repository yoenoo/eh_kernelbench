import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* x, int* out, int dim_size, int outer_size, int inner_size, int dim) {
    extern __shared__ byte[] block_shared;
    scalar_t* temp_val = (scalar_t*)block_shared;
    int* temp_idx = (int*)(temp_val + blockDim.x);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.x * dim_size;

    int outer = blockIdx.x / inner_size;
    int inner = blockIdx.x % inner_size;

    if (dim == 0) {
        outer = idx / dim_size;
        inner = idx % inner_size;
        offset = outer * dim_size + inner;
    } else if (dim == 1) {
        offset = blockIdx.x * dim_size;
        idx = threadIdx.x;
        outer = blockIdx.x / dim2;
        inner = blockIdx.x % dim2;
    } else if (dim == 2) {
        offset = blockIdx.x * inner_size + threadIdx.x;
    }

    scalar_t val = x[offset + idx];
    int arg = idx;

    temp_val[threadIdx.x] = val;
    temp_idx[threadIdx.x] = arg;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (temp_val[threadIdx.x] < temp_val[threadIdx.x + s]) {
                temp_val[threadIdx.x] = temp_val[threadIdx.x + s];
                temp_idx[threadIdx.x] = temp_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = temp_idx[0];
    }
}

torch::Tensor argmax_cuda(torch::Tensor x, int dim) {
    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    int out_size = 1;
    if (dim == 0) {
        out_size = dim1 * dim2;
    } else if (dim == 1) {
        out_size = batch_size * dim2;
    } else {
        out_size = batch_size * dim1;
    }

    auto out = torch::empty({out_size}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    dim3 threads(256);
    dim3 blocks(out_size);

    int shared_size = threads.x * sizeof(float) + threads.x * sizeof(int);

    argmax_kernel<float><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), out.data_ptr<int>(), 
        dim == 0 ? batch_size : dim == 1 ? dim1 : dim2,
        dim == 0 ? 1 : dim == 1 ? dim2 : 1,
        dim == 0 ? dim1 * dim2 : dim == 1 ? batch_size * dim2 : batch_size * dim1,
        dim
    );

    return out;
}
"""

argmax_cpp_source = (
    "torch::Tensor argmax_cuda(torch::Tensor x, int dim);"
)

# Compile the inline CUDA code for argmax
argmax = load_inline(
    name="argmax",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-D_FORCE_INLINE"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax = argmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax.argmax_cuda(x, self.dim).view(x.size()[:self.dim] + x.size()[self.dim+1:])