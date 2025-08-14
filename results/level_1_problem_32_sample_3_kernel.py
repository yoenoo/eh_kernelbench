import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardTanh
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_hardtanh_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = (val < -1.0f) ? -1.0f : (val > 1.0f ? 1.0f : val);
    }
}

torch::Tensor elementwise_hardtanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_hardtanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size
    );

    return out;
}
"""

hardtanh_cpp_header = "torch::Tensor elementwise_hardtanh_cuda(torch::Tensor x);"

# Compile the inline CUDA code
hardtanh = load_inline(
    name="hardtanh",
    cuda_sources=hardtanh_source,
    cpp_sources=hardtanh_cpp_header,
    functions=["elementwise_hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = hardtanh

    def forward(self, x):
        return self.hardtanh.elementwise_hardtanh_cuda(x)