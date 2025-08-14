cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Inline CUDA kernel for 1D average pooling
        avg_pool1d_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <vector>

        using tensor = torch::Tensor;

        template <typename scalar_t>
        __global__ void avg_pool1d_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int in_channels,
            const int input_length,
            const int output_length,
            const int kernel_size,
            const int padding_left,
            const int stride) {

            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * in_channels * output_length) return;

            const int output_x = idx % output_length;
            const int channel = (idx / output_length) % in_channels;
            const int batch = idx / (in_channels * output_length);

            int start_x = output_x * stride - padding_left;
            int end_x = start_x + kernel_size;
            scalar_t sum = 0;

            for (int i = start_x; i < end_x; i++) {
                if (i >=0 && i < input_length) {
                    sum += input[batch * in_channels * input_length + channel * input_length + i];
                }
            }

            output[idx] = sum / kernel_size;
        }

        tensor avg_pool1d_cuda(tensor input, int kernel_size, int stride, int padding) {
            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto input_length = input.size(2);

            int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

            auto output = torch::empty({batch_size, in_channels, output_length}, input.options());

            const int padding_left = padding;

            const int threads = 256;
            const int blocks = (batch_size * in_channels * output_length + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool1d_cuda", ([&] {
                avg_pool1d_kernel<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch_size,
                    in_channels,
                    input_length,
                    output_length,
                    kernel_size,
                    padding_left,
                    stride);
            }));

            return output;
        }
        """
        avg_pool1d_header = "torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

        self.avg_pool = load_inline(
            name='avg_pool1d',
            cpp_sources=avg_pool1d_header,
            cuda_sources=avg_pool1d_source,
            functions=['avg_pool1d_cuda'],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool.avg_pool1d_cuda(x, self.kernel_size, self.stride, self.padding)