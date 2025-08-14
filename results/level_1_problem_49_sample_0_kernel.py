import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

template <typename scalar_t>
__global__ void max_reduction_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out,
                                    int batch_size, int dim1, int dim2, int dim_to_reduce) {
    int batch_idx = blockIdx.x;
    int output_dim = (dim_to_reduce == 1 ? dim2 : dim1);
    int out_idx = batch_idx * output_dim + threadIdx.x;

    scalar_t max_val = -INFINITY;
    int x_idx_base = batch_idx * dim1 * dim2;
    if (dim_to_reduce == 1) {
        // Reducing over dim1 (4096), so iterate over dim1 and fix dim2 to threadIdx.x
        for (int i = 0; i < dim1; ++i) {
            int current_idx = x_idx_base + i * dim2 + threadIdx.x;
            scalar_t val = x[current_idx];
            if (val > max_val) {
                max_val = val;
            }
        }
    } else {
        // Reducing over dim2 (4095), so iterate over dim2 and fix dim1 to threadIdx.x
        for (int i = 0; i < dim2; ++i) {
            int current_idx = x_idx_base + threadIdx.x * dim2 + i;
            scalar_t val = x[current_idx];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    out[out_idx] = max_val;
}

std::tuple<torch::Tensor> max_reduction_cuda(torch::Tensor x, int dim) {
    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    int output_dim = (dim == 1 ? dim2 : dim1);
    auto out = torch::empty({batch_size, output_dim}, x.options());

    dim3 blocks(batch_size);
    dim3 threads(output_dim);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (dim == 1 || dim == 2) {
        AT_DISPATCH_ALL_TYPES(x.scalar_type(), "max_reduction_cuda", ([&] {
            max_reduction_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data<scalar_t>(), out.data<scalar_t>(), batch_size, dim1, dim2, dim);
        }));
    }

    return std::tuple<torch::Tensor>(out);
}
"""

max_reduction_cpp_source = """
std::tuple<torch::Tensor> max_reduction_cuda(torch::Tensor x, int dim);
"""

max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)[0]