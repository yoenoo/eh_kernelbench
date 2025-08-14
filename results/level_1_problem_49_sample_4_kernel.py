import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <atomic>

template <typename scalar_t>
__global__ void max_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const int batch_size,
    const int dim1,
    const int dim2,
    const int output_size,
    const int reduce_dim,
    const int outer_dim,
    const int inner_dim
) {
    extern __shared__ scalar_t shared_data[];

    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= output_size) return;

    scalar_t max_val = -INFINITY;

    const int batch_idx = output_index / (outer_dim * inner_dim);
    const int outer_idx = (output_index / inner_dim) % outer_dim;
    const int reduce_idx = threadIdx.x;

    scalar_t* shared_max = shared_data;

    // Initialize shared memory for each thread's value
    if (reduce_idx < inner_dim) {
        const int input_index = batch_idx * dim1 * dim2 +
                               (reduce_dim == 0 ? reduce_idx : outer_idx) * dim2 +
                               (reduce_dim == 0 ? outer_idx : reduce_idx);
        shared_max[threadIdx.x] = input[input_index];
    }

    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = max(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[output_index] = shared_max[0];
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    const int reduce_dim = dim;
    const int outer_dim = (dim == 0) ? 1 : (dim == 1 ? dim2 : dim1);
    const int inner_dim = (dim == 0) ? (dim1 * dim2) : (dim == 1 ? dim2 : dim1);
    const int output_size = batch_size * outer_dim;

    auto output = torch::empty({batch_size, outer_dim}, input.options());
    auto thread_count = inner_dim;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;

    dim3 blocks(num_blocks);
    dim3 threads(block_size);
    const int shared_size = block_size * sizeof(float);

    max_reduction_kernel<float><<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2,
        output_size,
        reduce_dim,
        outer_dim,
        inner_dim
    );

    cudaDeviceSynchronize();
    return output;
}
"""

max_reduction_cpp_source = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);
"""

max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduction_cuda = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction_cuda.max_reduction_cuda(x, self.dim)

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]