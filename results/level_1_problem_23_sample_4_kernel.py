import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_forward_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int num_rows,
                                      const int num_cols) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col >= num_cols) return;

    __shared__ scalar_t row_max;
    if (threadIdx.x == 0) {
        scalar_t max_val = input[row * num_cols];
        for (int i = 1; i < num_cols; ++i) {
            if (input[row * num_cols + i] > max_val) {
                max_val = input[row * num_cols + i];
            }
        }
        row_max = max_val;
    }
    __syncthreads();

    const scalar_t exp_x = exp(input[row * num_cols + col] - row_max);
    atomicAdd_block(&row_max, exp_x); // This is a hypothetical, not standard atomic. Need to reimplement with proper reduction.

    output[row * num_cols + col] = exp_x / row_max;
}

// Proper reduction using shared memory for sum calculation
template <typename scalar_t>
__global__ void softmax_forward_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int num_rows,
                                      const int num_cols) {
    extern __shared__ scalar_t shared[];

    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int col = blockIdx.x * blockDim.x + tid;

    scalar_t sum = 0;
    scalar_t max_val = input[row * num_cols + tid];

    // Load data into shared memory
    if (col < num_cols) {
        shared[tid] = input[row * num_cols + tid];
    }
    else {
        shared[tid] = -INFINITY;
    }

    // Compute max in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            if (shared[tid] < shared[tid + s]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }

    max_val = shared[0];
    __syncthreads();

    // Compute exp(x_i - max)
    scalar_t exp_x = 0;
    if (col < num_cols) {
        exp_x = exp(input[row * num_cols + col] - max_val);
    }
    shared[tid] = exp_x;

    // Sum the exp terms
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
    }
    sum = (tid == 0) ? shared[0] : 0;
    __syncthreads();

    if (col < num_cols) {
        output[row * num_cols + col] = exp_x / sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const dim3 blocks((dim + threads - 1) / threads, batch_size); // blockIdx.x is col blocks, blockIdx.y is row index

    // Dynamically request shared memory for maximum shared memory usage
    const size_t shared_mem = threads * sizeof(float) * 2; // space for exp and max
    softmax_forward_kernel<float><<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}

// Stub function declarations for compilation
torch::Tensor softmax_cuda_stub(torch::Tensor input) {
    return softmax_cuda(input);
}
"""

softmax_cuda_header = """
torch::Tensor softmax_cuda_stub(torch::Tensor input);
"""

softmax_ext = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cuda_header,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_cuda_stub"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda_stub(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []