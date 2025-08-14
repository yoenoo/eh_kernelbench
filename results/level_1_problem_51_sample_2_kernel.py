import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int* output,
                             int dim, int outer_dim, int inner_dim) {
    int batch_idx = blockIdx.x / inner_dim;
    int inner_idx = blockIdx.x % inner_dim;
    int idx = threadIdx.x;

    int input_idx = batch_idx * outer_dim * inner_dim + idx * inner_dim + inner_idx;
    int output_idx = batch_idx * inner_dim + inner_idx;

    __shared__ scalar_t shared_max;
    __shared__ int shared_argmax;

    if (threadIdx.x == 0) {
        shared_max = -INFINITY;
        shared_argmax = -1;
    }
    __syncthreads();

    if (idx < outer_dim) {
        scalar_t val = input[input_idx];
        int tid = threadIdx.x;
        // Using reduction to find max and its index
        for (int stride = 1; stride < outer_dim; stride *= 2) {
            __syncthreads();
            int pair_idx = tid + stride;
            if (pair_idx < outer_dim) {
                if (val < input[input_idx + stride * inner_dim]) {
                    val = input[input_idx + stride * inner_dim];
                }
            }
            __syncthreads();
        }
        // Now each thread has the max value in its segment
        if (tid == 0) {
            if (val > shared_max) {
                atomicMax(&shared_max, val);
                atomicExch(&shared_argmax, idx);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        output[output_idx] = shared_argmax;
    }
}

std::vector<int> compute_output_size(torch::Tensor input, int dim) {
    auto input_size = input.sizes().vec();
    input_size.erase(input_size.begin() + dim);
    return input_size;
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    auto output_size = compute_output_size(input, dim);
    torch::Tensor output = torch::empty(output_size, torch::dtype(torch::kInt32).device(input.device()));

    int batch_size = input.size(0);
    int outer_dim = input.size(dim);
    int inner_dim = 1;
    for (size_t i = 0; i < input.dim(); i++) {
        if (i != dim) {
            inner_dim *= input.size(i);
        }
    }
    inner_dim /= batch_size; // Assuming dim is 1 for the given problem's Model

    dim3 blocks(batch_size * inner_dim);
    dim3 threads(outer_dim); // Each block handles a single output element across threads

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data_ptr<int>(),
            dim,
            outer_dim,
            inner_dim
        );
    }));

    return output;
}
"""

argmax_cuda_header = """
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

argmax_op = load_inline(
    name="argmax_op",
    cpp_sources=argmax_cuda_header,
    cuda_sources=argmax_cuda_source,
    functions=["argmax_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_op = argmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_op.argmax_cuda(x, self.dim)