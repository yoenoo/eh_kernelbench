import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for parallel prefix sum (scan)
scan_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int BlockSize>
__global__ void inclusive_scan_kernel(const float* __restrict__ input, float* __restrict__ output,
                                    int batch_size, int dim_size, int dim_stride) {
    extern __shared__ float shared_data[];

    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    int global_offset = batch_id * dim_stride;

    // Load data into shared memory
    if (tid < dim_size) {
        shared_data[tid] = input[global_offset + tid];
    }
    __syncthreads();

    // Parallel scan in shared memory
    for (int offset = 1; offset < dim_size; offset <<= 1) {
        int index = 2 * offset * tid;
        if (index < dim_size && index + offset < dim_size) {
            shared_data[index + offset] += shared_data[index];
        }
        __syncthreads();
    }

    // Down-sweep to compute inclusive sum
    for (int offset = (dim_size >> 1); offset > 0; offset >>= 1) {
        int index = 2 * offset * tid;
        if (index + offset < dim_size) {
            float temp = shared_data[index];
            shared_data[index] = shared_data[index + offset] - temp;
            shared_data[index + offset] += temp; // Not necessary for inclusive but part of standard scan
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (tid < dim_size) {
        output[global_offset + tid] = shared_data[tid];
    }
}

torch::Tensor inclusive_scan_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim_size = input.size(dim);
    int dim_stride = input.stride(dim);

    auto output = torch::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(1024); // Adjust thread block size as needed
    int shared_mem = dim_size * sizeof(float);

    if (dim_size <= 1024) {
        threads.x = dim_size;
    } else {
        threads.x = 1024; // Choose maximum possible for large dimensions
    }

    inclusive_scan_kernel<1024><<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size,
        dim_stride
    );

    return output;
}
"""

cpp_wrapper = "torch::Tensor inclusive_scan_cuda(torch::Tensor input, int dim);"

# Compile the custom CUDA kernel
scan_op = load_inline(
    name='inclusive_scan',
    cpp_sources=[cpp_wrapper],
    cuda_sources=[scan_kernel],
    functions=['inclusive_scan_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scan_op = scan_op

    def forward(self, x):
        return self.scan_op.inclusive_scan_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]