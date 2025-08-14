cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void logsoftmax_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int batch_size, int dim_size, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < batch_size * dim_size; i += stride) {
        int n = i / dim_size;
        int d = i % dim_size;

        scalar_t max_val = -FLT_MAX;
        #pragma unroll
        for (int k = 0; k < dim_size; ++k) {
            if (k == d) {
                max_val = input[n * dim_size + k];
            } else {
                max_val = max(max_val, input[n * dim_size + k]);
            }
        }

        scalar_t sum = 0.0;
        #pragma unroll
        for (int k = 0; k < dim_size; ++k) {
            sum += exp(input[n * dim_size + k] - max_val);
        }

        output[i] = input[i] - max_val - log(static_cast<scalar_t>(sum));
    }
}

torch::Tensor logsoftmax_forward_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto dim_size = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (batch_size * dim_size + threads - 1) / threads;
    
    const int dim = 1; // Hardcoded dimension as per original model's default
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "logsoftmax_forward_cuda", ([&] {
        logsoftmax_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            dim
        );
    }));

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("logsoftmax_forward_cuda", &logsoftmax_forward_cuda, "LogSoftmax forward CUDA");
}
"""

logsoftmax_cpp_source = """
#include <torch/extension.h>
torch::Tensor logsoftmax_forward_cuda(torch::Tensor input);
"""

logsoftmax_extension = load_inline(
    name="logsoftmax",
    cpp Sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["logsoftmax_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.logsoftmax = logsoftmax_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax.logsoftmax_forward_cuda(x)