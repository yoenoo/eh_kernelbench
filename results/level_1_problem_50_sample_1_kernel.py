import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv2d_im2col_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void im2col_gpu_kernel(const int n, const scalar_t* data_im, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int h_out = index / kernel_h / kernel_w / width;
    int w_out = index / kernel_h / kernel_w % width;
    int k = index % (kernel_h * kernel_w);
    int h_k = k / kernel_w;
    int w_k = k % kernel_w;
    int h_in = h_out * stride_h - pad_h + h_k * dilation_h;
    int w_in = w_out * stride_w - pad_w + w_k * dilation_w;
    if (h_in < 0 || h_in >= height || w_in < 0 || w_in >= width) {
      data_col[index] = 0;
    } else {
      data_col[index] = data_im[h_in * width + w_in];
    }
  }
}

std::vector<int64_t> get_output_size(
    torch::Tensor input, 
    torch::Tensor weight, 
    int stride_h, 
    int stride_w, 
    int padding_h, 
    int padding_w, 
    int dilation_h, 
    int dilation_w) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    return {batch_size, out_channels, out_h, out_w};
}

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w) {

  auto output_size = get_output_size(input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
  int batch_size = output_size[0];
  int out_channels = output_size[1];
  int out_h = output_size[2];
  int out_w = output_size[3];

  auto input_reshaped = input.view({batch_size, -1, input.size(2), input.size(3)});
  int channels = input_reshaped.size(1);
  int height_col = input_reshaped.size(2);
  int width_col = input_reshaped.size(3);

  const int kernel_h = weight.size(2);
  const int kernel_w = weight.size(3);

  const int kernel_dim = channels * kernel_h * kernel_w;
  auto col_buffer = torch::empty({batch_size, kernel_dim, out_h * out_w}, 
                                torch::device(input.device()).dtype(input.scalar_type()));

  // Calculate im2col
  const int height = input_reshaped.size(2);
  const int width = input_reshaped.size(3);
  const int num_kernels = kernel_h * kernel_w;
  const int channels = input_reshaped.size(1);
  const int num_elements = channels * num_kernels * out_h * out_w;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "im2col_gpu", ([&] {
      im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
          num_elements,
          input_reshaped.data_ptr<scalar_t>(),
          height,
          width,
          kernel_h,
          kernel_w,
          padding_h,
          padding_w,
          stride_h,
          stride_w,
          dilation_h,
          dilation_w,
          col_buffer.data_ptr<scalar_t>());
  }));

  auto weight_reshaped = weight.view({-1, kernel_dim});
  auto bias_reshaped = bias.unsqueeze(1);

  auto output = torch::addmm(bias_reshaped, weight_reshaped, col_buffer.view({kernel_dim, -1}));
  return output.view(output_size);
}

TORCH_LIBRARY(my_ops, m) {
  m.def("custom_conv2d", &conv2d_forward, 
    "Input: Tensor input, Tensor weight, Tensor bias, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv2d", &conv2d_forward, "Custom Conv2d Forward");
}
"""

cpp_conv_source = """
#include <torch/extension.h>
torch::Tensor custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w);
"""

# Compile the custom convolution kernel
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=cpp_conv_source,
    cuda_sources=conv2d_im2col_source,
    functions=["custom_conv2d"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(96, 3, 11, 11))  # Custom Conv2d parameters
        self.bias = nn.Parameter(torch.randn(96))               # Custom bias
        self.stride = (4, 4)
        self.padding = (2, 2)
        self.dilation = (1, 1)
        self.custom_conv = custom_conv2d

    def forward(self, x):
        # Extract parameters
        weight = self.weight
        bias = self.bias
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        x = self.custom_conv.custom_conv2d(
            x, 
            weight, 
            bias, 
            stride_h, 
            stride_w, 
            padding_h, 
            padding_w,
            dilation_h,
            dilation_w
        )
        return x