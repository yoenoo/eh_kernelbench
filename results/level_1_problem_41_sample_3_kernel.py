import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA MaxPool1d kernel
max_pool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool_1d_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  int64_t batch_size,
                                  int64_t num_features,
                                  int64_t in_length,
                                  int64_t out_length,
                                  int kernel_size,
                                  int stride,
                                  int padding,
                                  int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y;
    int idz = blockIdx.z;

    if (idx >= out_length || idy >= num_features || idz >= batch_size) {
        return;
    }

    const int in_index_start = idx * stride - padding;
    scalar_t max_val = -FLT_MAX;
    int max_idx = -1;
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = in_index_start + k * dilation;
        if (in_pos < 0 || in_pos >= in_length) {
            continue;
        }
        scalar_t val = input[idz * num_features * in_length + idy * in_length + in_pos];
        if (val > max_val) {
            max_val = val;
            max_idx = in_pos;
        }
    }
    output[idz * num_features * out_length + idy * out_length + idx] = max_val;
}

torch::Tensor max_pool_1d_cuda(torch::Tensor input,
                              int kernel_size,
                              int stride,
                              int padding,
                              int dilation) {
    const auto batch_size = input.size(0);
    const auto num_features = input.size(1);
    const auto in_length = input.size(2);
    const auto out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, num_features, out_length}, input.options());

    int threads = 256;
    dim3 blocks(out_length, num_features, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool_1d_cuda", ([&] {
        max_pool_1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            num_features,
            in_length,
            out_length,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    return output;
}
"""

max_pool_1d_cpp_source = """
torch::Tensor max_pool_1d_cuda(torch::Tensor input,
                              int kernel_size,
                              int stride,
                              int padding,
                              int dilation);
"""

# Compile the custom CUDA kernel
max_pool_1d = load_inline(
    name="max_pool_1d",
    cpp_sources=[max_pool_1d_cpp_source],
    cuda_sources=[max_pool_1d_source],
    functions=["max_pool_1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.max_pool_1d = max_pool_1d

    def forward(self, x):
        # The original PyTorch implementation returns both values and indices when return_indices is True
        # However, our custom kernel currently only computes the output values
        # So we'll raise an error if return_indices is requested
        assert not self.return_indices, "Custom kernel does not support return_indices=True"

        output = self.max_pool_1d.max_pool_1d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        return output