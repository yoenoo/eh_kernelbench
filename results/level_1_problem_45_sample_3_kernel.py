import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        # Define and compile the custom CUDA kernel
        avg_pool2d_source = f"""
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void avg_pool2d_kernel(
            scalar_t* __restrict__ output,
            scalar_t* __restrict__ input,
            int batch_size, int channels, int in_height, int in_width,
            int out_height, int out_width,
            int kernel_h, int kernel_w,
            int pad_h, int pad_w,
            int stride_h, int stride_w) {{
                int batch_idx = blockIdx.x;
                int channel_idx = blockIdx.y;
                int out_y = threadIdx.y;
                int out_x = threadIdx.x;

                int in_y = -pad_h + out_y * stride_h;
                int in_x = -pad_w + out_x * stride_w;

                scalar_t sum = 0;
                for (int ky = 0; ky < kernel_h; ++ky) {{
                    for (int kx = 0; kx < kernel_w; ++kx) {{
                        int y = in_y + ky;
                        int x = in_x + kx;
                        if (y >=0 && y < in_height && x >=0 && x < in_width) {{
                            sum += input[batch_idx * channels * in_height * in_width +
                                        channel_idx * in_height * in_width +
                                        y * in_width + x];
                        }}
                    }}
                }}
                output[batch_idx * channels * out_height * out_width +
                      channel_idx * out_height * out_width +
                      out_y * out_width + out_x] = sum / (kernel_h * kernel_w);
            }}

        void avg_pool2d_cuda(torch::Tensor input, torch::Tensor output,
                            int kernel_h, int kernel_w,
                            int pad_h, int pad_w,
                            int stride_h, int stride_w) {{
            int batch_size = input.size(0);
            int channels = input.size(1);
            int in_height = input.size(2);
            int in_width = input.size(3);
            int out_height = output.size(2);
            int out_width = output.size(3);

            dim3 threads(out_width, out_height);
            dim3 blocks(batch_size, channels);
            AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool2d_cuda", ([&] {{
                avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    batch_size, channels, in_height, in_width,
                    out_height, out_width,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w);
            }}));
        }}
        """
        avg_pool2d_h_source = """
        void avg_pool2d_cuda(torch::Tensor input, torch::Tensor output,
            int kernel_h, int kernel_w, int pad_h, int pad_w,
            int stride_h, int stride_w);
        """

        self.avg_pool_cuda = load_inline(
            name=f"avg_pool2d_{kernel_size}",
            cpp_sources=avg_pool2d_h_source,
            cuda_sources=avg_pool2d_source,
            functions=["avg_pool2d_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, in_height, in_width = x.shape
        kernel_h = self.kernel_size
        kernel_w = self.kernel_size
        stride_h = self.stride
        stride_w = self.stride
        pad_h = self.padding
        pad_w = self.padding

        # Calculate output dimensions
        out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
        out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

        output = torch.empty(batch_size, channels, out_height, out_width, device=x.device)

        self.avg_pool_cuda.avg_pool2d_cuda(
            x,
            output,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w
        )

        return output