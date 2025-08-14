import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool1d_kernel(const scalar_t* __restrict__ input,
                                scalar_t* __restrict__ output,
                                const int batch_size,
                                const int num_channels,
                                const int in_length,
                                const int kernel_size,
                                const int stride,
                                const int padding,
                                const int dilation,
                                const int out_length) {
    const int idx_batch = blockIdx.x;
    const int idx_channel = blockIdx.y;
    const int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_pos >= out_length) return;

    int in_start = out_pos * stride - padding;
    int max_val = -INFINITY;
    int max_idx = 0;

    for (int i = 0; i < kernel_size; ++i) {
        int in_pos = in_start + i * dilation;
        if (in_pos < 0 || in_pos >= in_length) {
            continue;
        }
        scalar_t val = input[idx_batch * num_channels * in_length +
                            idx_channel * in_length + in_pos];
        if (val > max_val) {
            max_val = val;
            max_idx = in_pos;
        }
    }

    output[idx_batch * num_channels * out_length +
           idx_channel * out_length + out_output] = max_val;
}

torch::Tensor max_pool1d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation,
                             bool return_indices) {
    const auto batch_size = input.size(0);
    const auto num_channels = input.size(1);
    const auto in_length = input.size(2);
    
    // Compute output length
    const int in_length_padded = in_length + 2 * padding;
    const int kernel_total = dilation * (kernel_size - 1) + 1;
    const int effective_stride = stride == 0 ? kernel_size : stride;
    const int out_length = (in_length_padded - kernel_total) / effective_stride + 1;

    auto options = torch::TensorOptions().like(input);
    torch::Tensor output = torch::empty({batch_size, num_channels, out_length}, options);

    dim3 threads(1024);
    dim3 blocks(batch_size, num_channels, out_length);

    int block_size = 1024;
    if (threads.x > out_length) {
        block_size = out_length;
        threads.x = block_size;
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool1d_cuda", ([&]{
        max_pool1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_channels,
            in_length,
            kernel_size,
            stride,
            padding,
            dilation,
            out_length);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

maxpool1d_cpp_source = """
torch::Tensor max_pool1d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation,
                             bool return_indices);
"""

max_pool1d = load_inline(name='max_pool1d',
                        cuda_sources=maxpool1d_source,
                        cpp_sources=maxpool1d_cpp_source,
                        functions=['max_pool1d_cuda'],
                        verbose=True)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0,
                 dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool = max_pool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)