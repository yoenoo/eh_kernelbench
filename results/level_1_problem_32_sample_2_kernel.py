import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val < min_val) {
            output[idx] = min_val;
        } else if (val > max_val) {
            output[idx] = max_val;
        } else {
            output[idx] = val;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto output = torch::empty_like(input);
    auto size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        min_val,
        max_val
    );

    return output;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);"

hardtanh_extension = load_inline(
    name="hardtanh_extension",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.min_val = -1.0
        self.max_val = 1.0
        self.hardtanh_cuda = hardtanh_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh_cuda.hardtanh_cuda(x, self.min_val, self.max_val)