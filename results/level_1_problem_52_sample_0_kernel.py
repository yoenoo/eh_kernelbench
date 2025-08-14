import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin operation
argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* input, int64_t* output,
                             int batch_size, int dim1, int dim2, int dim) {
    int batch_idx = blockIdx.x;
    int idx = threadIdx.x;
    int stride = 1;
    if (dim == 0) {
        stride = dim1;
    } else if (dim == 1) {
        stride = dim2;
    }

    extern __shared__ intshared_t block_data[];
    scalar_t* sdata = (scalar_t*)block_data;
    int64_t* s_indices = (int64_t*)(sdata + blockDim.x);

    // Load data into shared memory
    int global_idx = batch_idx * dim1 * dim2 + idx * stride;
    if (idx < stride) {
        sdata[idx] = input[global_idx];
        s_indices[idx] = idx;
    } else {
        sdata[idx] = std::numeric_limits<scalar_t>::max();
        s_indices[idx] = 0;
    }
    __syncthreads();

    // Perform parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            if (sdata[idx] > sdata[idx + s]) {
                sdata[idx] = sdata[idx + s];
                s_indices[idx] = s_indices[idx + s];
            }
        }
        __syncthreads();
    }

    if (idx == 0) {
        output[batch_idx * dim2 + (dim == 0 ? idx : (batch_idx * dim1 + idx))] = s_indices[0];
    }
}

std::tuple<torch::Tensor> argmin_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);
    auto output = torch::empty({batch_size, dim == 0 ? dim1 : dim2}, input.options().dtype(torch::kLong));

    int block_size = 256;
    int shared_size = 2 * block_size * sizeof(float) + block_size * sizeof(int64_t);
    dim3 blocks(batch_size);
    dim3 threads(block_size);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t><<<blocks, threads, shared_size, torch::cuda::current_stream()>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<int64_t>(),
            batch_size, dim1, dim2, dim);
    }));

    return output;
}
"""

argmin_cpp_source = """
std::tuple<torch::Tensor> argmin_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for argmin
argmin = load_inline(
    name="argmin_op",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.argmin_op = argmin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin_op.argmin_cuda(x, self.dim)[0]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x.cuda()]

def get_init_inputs():
    return [dim]