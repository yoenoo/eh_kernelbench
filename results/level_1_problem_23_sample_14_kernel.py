import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ input, int batch_size, int dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
        const auto* input_row = input + tid * dim;
        scalar_t* output_row = output + tid * dim;

        // Find the maximum value for numerical stability
        scalar_t max_val = -INFINITY;
        for (int i = 0; i < dim; ++i) {
            max_val = max(max_val, input_row[i]);
        }

        scalar_t sum = 0.0;
        // Subtract max and exponentiate
        for (int i = 0; i < dim; ++i) {
            scalar_t e = exp(input_row[i] - max_val);
            output_row[i] = e;
            sum += e;
        }

        // Normalize
        scalar_t inv_sum = 1.0 / sum;
        for (int i = 0; i < dim; ++i) {
            output_row[i] *= inv_sum;
        }
    }
}

template <typename scalar_t>
torch::Tensor _softmax_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;

    softmax_kernel<scalar_t><<<grid_size, block_size>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        batch_size,
        dim
    );

    return output;
}

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    if (input.dtype() == torch::kFloat32) {
        return _softmax_cuda<float>(input);
    } else if (input.dtype() == torch::kFloat16) {
        return _softmax_cuda<__half>(input);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}
"""

softmax_cpp_source = """
torch::Tensor softmax_forward_cuda(torch::Tensor input);
"""

softmax_ext = load_inline(
    name="softmax_extension",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_kernel_source,
    functions=["softmax_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_forward_cuda(x.cuda())