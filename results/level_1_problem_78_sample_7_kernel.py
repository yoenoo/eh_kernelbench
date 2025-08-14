import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int input_h,
    const int input_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int out_h,
    const int out_w) {
  
  CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * out_h * out_w) {
    int w_out = index % out_w;
    int h_out = (index / out_w) % out_h;
    int c_out = (index / (out_w * out_h)) % out_channels;
    int n = index / (out_channels * out_h * out_w);

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_in = (h_out - kh) / stride_h;
        const int w_in = (w_out - kw) / stride_w;
        // Check if the current position is within input boundaries
        if ((h_out - kh) % stride_h == 0 && 
            (w_out - kw) % stride_w == 0 &&
            h_in >= -pad_h && h_in < input_h + pad_h &&
            w_in >= -pad_w && w_in < input_w + pad_w) {
          int ic_offset = (h_in + pad_h) * input_w + (w_in + pad_w);
          for (int c_in = 0; c_in < in_channels; ++c_in) {
            const scalar_t w_val = weight[(c_out * in_channels + c_in) * kernel_h * kernel_w + kh * kernel_w + kw];
            const scalar_t in_val = input[n * in_channels * input_h * input_w + 
                                          c_in * input_h * input_w + 
                                          ic_offset];
            val += w_val * in_val;
          }
        }
      }
    }
    output[index] = val;
  }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_h,
                                   int kernel_w,
                                   int stride_h,
                                   int stride_w,
                                   int pad_h,
                                   int pad_w) {
  
  const auto batch_size = input.size(0);
  const auto in_channels = input.size(1);
  const auto out_channels = weight.size(0);
  const auto input_h = input.size(2);
  const auto input_w = input.size(3);

  const auto out_h = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
  const auto out_w = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;

  auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

  const int blocks = 512;
  const int threads = 1024;
  const int total_threads = blocks * threads;

  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_CUDA_CHECK(cudaDeviceSynchronize());

  conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
      input.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      batch_size,
      in_channels,
      out_channels,
      kernel_h,
      kernel_w,
      input_h,
      input_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      out_h,
      out_w);

  return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_h,
                                   int kernel_w,
                                   int stride_h,
                                   int stride_w,
                                   int pad_h,
                                   int pad_w);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose2d.conv_transpose2d_cuda(
            x.cuda(),
            self.weight,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output