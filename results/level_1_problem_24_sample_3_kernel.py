cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void logsoftmax_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, const int batch_size, const int dim_size, const int dim) {
    const int batch_stride = dim_size;
    const int batch_id = blockIdx.x;
    const int element_id = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (element_id >= dim_size) return;

    int idx = batch_id * batch_stride + element_id;
    
    // Subtract max for numerical stability
    scalar_t max_val = -INFINITY;
    #pragma unroll
    for (int i = 0; i < dim_size; ++i) {
        scalar_t val = input[batch_id * batch_stride + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    scalar_t sum_exp = 0;
    #pragma unroll
    for (int i = 0; i < dim_size; ++i) {
        scalar_t val = input[batch_id * batch_stride + i] - max_val;
        sum_exp += exp(val);
    }

    scalar_t logsoftmax_val = (input[idx] - max_val) - log(sum_exp);
    output[idx] = logsoftmax_val;
}

std::tuple<torch::Tensor, torch::Tensor> logsoftmax_forward_cuda(torch::Tensor input, int64_t dim) {
    const auto batch_size = input.size(0);
    const auto dim_size = input.size(dim);
    const auto total_elements = input.numel();
    
    auto output = torch::empty_like(input);
    auto max_values = torch::empty({batch_size}, input.options());

    dim3 threads(256, 1);
    dim3 blocks(batch_size, dim_size);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "logsoftmax_forward_cuda", ([&] {
        logsoftmax_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            dim);
    }));

    return std::make_tuple(output, max_values);
}

// TODO: Implement backward kernel here

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("logsoftmax_forward", &logsoftmax_forward_cuda, "LogSoftmax forward");
  // m.def("logsoftmax_backward", &logsoftmax_backward_cuda, "LogSoftmax backward");
}
"""

cpp_source = """
#include <torch/extension.h>
"""

logsoftmax_ext = load_inline(
    name="logsoftmax_ext",
    cpp_sources=cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["logsoftmax_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = logsoftmax_ext.logsoftmax_forward(x, self.dim)
        return output