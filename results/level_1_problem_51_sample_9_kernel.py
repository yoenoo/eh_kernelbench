import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ int argmax_cuda(const scalar_t* data, int len) {
    int max_idx = 0;
    scalar_t max_val = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

__global__ void argmax_kernel(const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input,
                             torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> output,
                             int dim) {
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= input.size(0)) return;
    
    if (dim == 1) {
        for (int i = 0; i < input.size(2); i++) {
            output[outer][i] = argmax_cuda(input[outer][i], input.size(1));
        }
    } else {
        for (int i = 0; i < input.size(1); i++) {
            output[outer][i] = argmax_cuda(input[outer][i], input.size(2));
        }
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    const int threads = 256;
    const int blocks = (input.size(0) + threads - 1) / threads;

    auto output = torch::empty({input.size(0), dim == 1 ? input.size(2) : input.size(1)}, 
                              torch::device(input.device()).dtype(torch::kInt32));

    argmax_kernel<<<blocks, threads>>>(
        input.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        output.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        dim
    );

    return output;
}
"""

argmax_kernel_cpp = """
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

argmax_op = load_inline(name="argmax_op",
                       cpp_sources=argmax_kernel_cpp,
                       cuda_sources=argmax_kernel_source,
                       functions=["argmax_cuda"],
                       verbose=True)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_op = argmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_op.argmax_cuda(x, self.dim)