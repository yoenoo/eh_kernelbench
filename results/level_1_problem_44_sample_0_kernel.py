import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        avg_pool_1d_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <ATen/ATen.h>

        template <typename scalar_t>
        __global__ void avg_pool_1d_forward_kernel(
            const torch::PackedTensorAccessor<scalar_t,3> input,
            torch::PackedTensorAccessor<scalar_t,3> output,
            const int batch_size,
            const int channels,
            const int input_length,
            const int output_length,
            const int kernel_size,
            const int padding,
            const int stride) {

            const int batch_id = blockIdx.x;
            const int channel_id = blockIdx.y;
            const int out_pos = threadIdx.x;

            scalar_t sum = 0;
            scalar_t count = 0;

            const int start = out_pos * stride - padding;
            const int end = start + kernel_size;

            for (int i = start; i < end; i++) {
                if (i >= 0 && i < input_length) {
                    sum += input[batch_id][channel_id][i];
                    count += 1;
                }
            }
            output[batch_id][channel_id][out_pos] = sum / count;
        }

        at::Tensor avg_pool_1d_forward_cuda(
            at::Tensor input,
            int kernel_size,
            int stride,
            int padding) {

            const auto batch_size = input.size(0);
            const auto channels = input.size(1);
            const auto input_length = input.size(2);
            const auto output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

            auto output = at::empty({batch_size, channels, output_length}, input.options());

            dim3 grid(batch_size, channels);
            dim3 block(output_length);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool_1d_forward", ([&] {
                avg_pool_1d_forward_kernel<scalar_t><<<grid, block>>>(
                    input.packed_accessor<scalar_t,3>(),
                    output.packed_accessor<scalar_t,3>(),
                    batch_size,
                    channels,
                    input_length,
                    output_length,
                    kernel_size,
                    padding,
                    stride);
            }));

            return output;
        }
        """

        avg_pool_1d = load_inline(
            name="avg_pool_1d",
            cpp_sources="",
            cuda_sources=avg_pool_1d_source,
            functions=["avg_pool_1d_forward_cuda"],
            verbose=True
        )

        self.forward_op = avg_pool_1d.avg_pool_1d_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_op(x, self.kernel_size, self.stride, self.padding)