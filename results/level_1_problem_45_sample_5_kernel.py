cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        # Load custom CUDA kernel
        avg_pool_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void avg_pool2d_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            int batch_size,
            int channels,
            int input_height,
            int input_width,
            int output_height,
            int output_width,
            int kernel_h,
            int kernel_w,
            int stride_h,
            int stride_w,
            int padding_h,
            int padding_w) {
            int batch_idx = blockIdx.x;
            int channel_idx = blockIdx.y;
            int out_y = threadIdx.y;
            int out_x = threadIdx.x;

            scalar_t sum = 0.0;
            int count = 0;

            int in_x_start = out_x * stride_w - padding_w;
            int in_y_start = out_y * stride_h - padding_h;

            for (int ky = 0; ky < kernel_h; ++ky) {
                int in_y = in_y_start + ky;
                if (in_y < 0 || in_y >= input_height)
                    continue;
                for (int kx = 0; kx < kernel_w; ++kx) {
                    int in_x = in_x_start + kx;
                    if (in_x < 0 || in_x >= input_width)
                        continue;
                    sum += input[batch_idx * channels * input_height * input_width +
                                channel_idx * input_height * input_width +
                                in_y * input_width + in_x];
                    count++;
                }
            }
            output[batch_idx * channels * output_height * output_width +
                   channel_idx * output_height * output_width +
                   out_y * output_width + out_x] = sum / count;
        }

        at::Tensor avg_pool2d_cuda(at::Tensor input, int kernel_size, int stride_h, int stride_w, int padding_h, int padding_w) {
            const auto batch_size = input.size(0);
            const auto channels = input.size(1);
            const auto input_height = input.size(2);
            const auto input_width = input.size(3);
            const auto kernel_h = kernel_size;
            const auto kernel_w = kernel_size;
            const auto stride_h_val = stride_h;
            const auto stride_w_val = stride_w;
            const auto padding_h_val = padding_h;
            const auto padding_w_val = padding_w;

            int output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
            int output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;

            auto output = at::empty({batch_size, channels, output_height, output_width}, input.options());

            dim3 block_dim(output_width, output_height, 1);
            dim3 grid_dim(batch_size, channels, 1);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool2d_cuda", ([&]{
                avg_pool2d_kernel<scalar_t><<<grid_dim, block_dim>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    kernel_h,
                    kernel_w,
                    stride_h_val,
                    stride_w_val,
                    padding_h_val,
                    padding_w_val);
            }));

            return output;
        }
        """

        avg_pool_cpp_source = """
        at::Tensor avg_pool2d_cuda(at::Tensor input, int kernel_size, int stride_h, int stride_w, int padding_h, int padding_w);
        """
        self.avg_pool_cuda = load_inline(
            name="avg_pool_cuda",
            cpp_sources=avg_pool_cpp_source,
            cuda_sources=avg_pool_source,
            functions=["avg_pool2d_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool2d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding
        )

def get_inputs():
    batch_size = 16
    channels = 64
    height = 2048
    width = 2048
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    kernel_size = 11
    return [kernel_size]