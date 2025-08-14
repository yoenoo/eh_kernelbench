cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int* output,
                             int batch_size, int dim1, int dim2, int dim) {
    const int outer_dim = dim == 0 ? 1 : dim1;
    const int inner_dim = dim == 0 ? dim1 * dim2 : dim2;
    const int output_rows = batch_size * (dim == 0 ? dim1 * dim2 : (dim == 1 ? dim2 : dim1));
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < output_rows;
         index += blockDim.x * gridDim.x) {
        int outer = index / inner_dim;
        int inner = index % inner_dim;
        int max_val = -1;
        int max_idx = -1;
        for (int i = 0; i < outer_dim; ++i) {
            int input_idx = 0;
            if (dim == 0) {
                input_idx = (i * dim1 * dim2) + outer * dim2 + inner;
            } else if (dim == 1) {
                input_idx = (outer * dim2 * outer_dim) + i * dim2 + inner;
            } else { // dim == 2
                input_idx = (outer * dim1 * dim2) + i * dim1 + inner;
            }
            if (input[input_idx] > max_val) {
                max_val = input[input_idx];
                max_idx = i;
            }
        }
        output[index] = max_idx;
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);
    auto output_size = input.sizes().vec();
    output_size.erase(output_size.begin() + dim);
    torch::Tensor output = torch::empty(output_size, input.options().dtype(torch::kInt32).device(torch::kCUDA));

    int threads = 256;
    int elements = output.numel();
    int blocks = (elements + threads - 1) / threads;

    // The kernel is launched with dynamic parallelism based on the output size
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data_ptr<int>(),
            batch_size,
            dim1,
            dim2,
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmax_cuda_header = """
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

argmax_cuda = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cuda_header,
    cuda_sources=argmax_cuda_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_cuda_op = argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return self.argmax_cuda_op.argmax_cuda(x, self.dim).to(x.device)
        else:
            return torch.argmax(x, dim=self.dim)