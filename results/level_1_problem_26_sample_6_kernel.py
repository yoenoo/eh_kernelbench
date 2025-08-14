import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        const float a = 0.7978845608f;  // sqrt(2/pi) ≈ 0.7978845608
        const float b = 0.044715f;
        float tmp = xi + b * xi * xi * xi;
        float tanh_val = tanhf(a * tmp);
        y[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    return y;
}
"""

gelu_cpp_source = """
extern "C" TORCH_API torch::Tensor gelu_cuda(torch::Tensor x);
"""

gelu_module = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_cuda = gelu_module.gelu_cuda

    def forward(self, x):
        return self.gelu_cuda(x)