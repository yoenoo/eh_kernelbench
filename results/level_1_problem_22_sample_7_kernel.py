import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Tanh activation
tanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    int64_t size = input.numel();
    torch::Tensor output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();  // Ensure kernel completes

    return output;
}
"""

tanh_cuda_header = "torch::Tensor tanh_cuda(torch::Tensor input);"

# Load the CUDA kernel
tanh_cuda = load_inline(
    name="tanh_cuda",
    cpp_sources=tanh_cuda_header,
    cuda_sources=tanh_cuda_source,
    functions=["tanh_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_cuda = tanh_cuda  # Reference to the loaded module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda.tanh_cuda(x)
        
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()  # Move input to CUDA
    return [x]

def get_init_inputs():
    return []