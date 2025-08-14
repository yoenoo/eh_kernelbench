import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        out[idx] = xi / (1.0f + expf(-xi));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

swish_cpp_header = """
torch::Tensor swish_cuda(torch::Tensor x);
"""

swish_cuda = load_inline(
    name="swish_cuda",
    cpp_sources=swish_cpp_header,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish_cuda = swish_cuda

    def forward(self, x):
        return self.swish_cuda.swish_cuda(x)