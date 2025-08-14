import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        avg_pool1d_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void avg_pool1d_forward_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int channels,
            const int in_length,
            const int out_length,
            const int kernel_size,
            const int padding,
            const int stride) {

            const int batch_idx = blockIdx.x;
            const int channel_idx = blockIdx.y;
            const int out_pos = threadIdx.x;

            const int in_start = out_pos * stride - padding;
            const int in_end = in_start + kernel_size;
            const int out_offset = batch_idx * channels * out_length + channel_idx * out_length + out_pos;

            scalar_t sum = 0;
            for (int i = in_start; i < in_end; ++i) {
                if (i >= 0 && i < in_length) {
                    sum += input[batch_idx * channels * in_length + channel_idx * in_length + i];
                }
            }
            output[out_offset] = sum / kernel_size;
        }

        torch::Tensor avg_pool1d_forward(torch::Tensor input) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int in_length = input.size(2);
            const int out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

            torch::Tensor output = torch::empty({batch_size, channels, out_length}, 
                                                dtype(input.dtype()), 
                                                device(input.device()));

            dim3 blocks(batch_size, channels);
            dim3 threads(out_length);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool1d_forward", ([&] {
                avg_pool1d_forward_kernel<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch_size,
                    channels,
                    in_length,
                    out_length,
                    kernel_size,
                    padding,
                    stride);
            }));

            return output;
        }
        """
        
        self.avg_pool = load_inline(
            name="avg_pool1d",
            cpp_sources="",
            cuda_sources=avg_pool1d_source,
            functions=["avg_pool1d_forward"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool.avg_pool1d_forward(x)