import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for parallel prefix sum (scan)
scan_cuda_src = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void exclusiveScanKernel(scalar_t *data, scalar_t *result, const int rows, const int cols, const int dim_size)
{
    __shared__ scalar_t s_data[2 * 1024]; // Shared memory buffer per block

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    scalar_t *block_data = data + row * cols;
    scalar_t *block_result = result + row * cols;

    // Load data into shared memory
    if (tid < cols) {
        s_data[tid] = block_data[tid];
    }
    __shared__ int offset;
    if (tid == 0) {
        s_data[cols] = 0; // Initialize with zero for exclusive scan
        offset = 0;
    }
    __syncthreads();

    // Up-sweep phase
    for (int d = 1; d < cols; d *= 2) {
        int index = 2 * d * tid;
        if (index < cols) {
            s_data[index + 2*d - 1] += s_data[index + d - 1];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int d = cols/2; d > 0; d /= 2) {
        if (tid < d) {
            int i = 2*d*(tid+1)-1;
            s_data[i] += s_data[i - d];
        }
        __syncthreads();
    }

    // Generate scan
    if (tid < cols) {
        block_result[tid] = s_data[tid] + offset;
    }
}

torch::Tensor exclusive_scan_cuda(torch::Tensor input, int dim)
{
    const int rows = input.size(0);
    const int cols = input.size(1);
    const int threadsPerBlock = 1024;

    auto output = torch::empty_like(input);

    dim3 blocks(rows);
    dim3 threads(threadsPerBlock);

    // Kernel selection based on data type
    AT_DISPATCH_FLOATING_TYPES(input.type(), "exclusive_scan_cuda", ([&] {
        exclusiveScanKernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols,
            cols);
    }));

    cudaDeviceSynchronize();
    return output;
}

"""

scan_cpp_src = """
torch::Tensor exclusive_scan_cuda(torch::Tensor input, int dim);
"""

# Compile the CUDA kernel
exclusive_scan_cuda = load_inline(
    name='exclusive_scan_cuda',
    cpp_sources=scan_cpp_src,
    cuda_sources=scan_cuda_src,
    functions=['exclusive_scan_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cuda_module = exclusive_scan_cuda

    def forward(self, x):
        if self.dim != 1:
            raise NotImplementedError("Currently supports only dim=1 for CUDA kernel.")
        return self.cuda_module.exclusive_scan_cuda(x, self.dim)