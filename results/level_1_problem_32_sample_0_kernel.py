import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_forward_kernel(const float* input, float* output, int n_elements, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float val = input[idx];
        if (val < min_val) val = min_val;
        else if (val > max_val) val = max_val;
        output[idx] = val;
    }
}

torch::Tensor hardtanh_forward_cuda(torch::Tensor input, float min_val, float max_val) {
    int n_elements = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;

    hardtanh_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements,
        min_val,
        max_val
    );

    return output;
}
"""

hardtanh_header = "torch::Tensor hardtanh_forward_cuda(torch::Tensor input, float min_val, float max_val);"

hardtanh_extension = load_inline(
    name="hardtanh_ext",
    cpp_sources=hardtanh_header,
    cuda_sources=hardtanh_kernel_source,
    functions=["hardtanh_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh_forward = hardtanh_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh_forward.hardtanh_forward_cuda(x, -1.0, 1.0)