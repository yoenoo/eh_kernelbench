import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ static inline float sigmoid_fma(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void custom_sigmoid_kernel(float* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = sigmoid_fma(in[idx]);
    }
}

torch::Tensor custom_sigmoid_cuda(torch::Tensor in) {
    auto out = torch::empty_like(in);
    int size = in.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    custom_sigmoid_kernel<<<num_blocks, block_size>>>((float*)out.data_ptr(), (const float*)in.data_ptr(), size);
    return out;
}
"""

sigmoid_cpp_source = "torch::Tensor custom_sigmoid_cuda(torch::Tensor in);"

custom_sigmoid = load_inline(
    name="custom_sigmoid",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["custom_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid_op = custom_sigmoid

    def forward(self, x):
        return self.sigmoid_op.custom_sigmoid_cuda(x)