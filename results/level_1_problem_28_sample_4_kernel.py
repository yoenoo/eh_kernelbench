import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val <= -3) {
            y[idx] = 0.0;
        } else if (val >= 3) {
            y[idx] = 1.0;
        } else {
            y[idx] = (val / 6.0f) + 0.5f;
        }
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

hardsigmoid_cpp_source = "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"

# Compile the inline CUDA code for HardSigmoid
hardsigmoid_op = load_inline(
    name="hardsigmoid_op",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_cuda = hardsigmoid_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid_cuda.hardsigmoid_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []