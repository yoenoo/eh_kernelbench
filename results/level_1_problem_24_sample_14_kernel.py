import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

template <typename scalar_t>
__global__ void logsoftmax_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int batch_size, int dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        for (int d = 0; d < dim_size; d++) {
            scalar_t val = input[idx * dim_size + d];
            if (val > max_val) {
                max_val = val;
            }
        }
        scalar_t sum = 0;
        for (int d = 0; d < dim_size; d++) {
            sum += exp(input[idx * dim_size + d] - max_val);
        }
        scalar_t inv_sum = 1.0 / sum;
        for (int d = 0; d < dim_size; d++) {
            output[idx * dim_size + d] = log(sum) + max_val;
            output[idx * dim_size + d] = input[idx * dim_size + d] - output[idx * dim_size + d];
        }
    }
}

torch::Tensor logsoftmax_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    const int shared_mem = 0;
    // Launch kernel
    logsoftmax_forward_kernel<float><<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size
    );
    return output;
}
"""

logsoftmax_cpp_source = """
torch::Tensor logsoftmax_forward_cuda(torch::Tensor input);
"""

logsoftmax_cuda = load_inline(
    name="logsoftmax_cuda",
    cpp_sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["logsoftmax_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.logsoftmax_cuda = logsoftmax_cuda
    
    def forward(self, x):
        return self.logsoftmax_cuda.logsoftmax_forward_cuda(x)