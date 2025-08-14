import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from torch.cuda.amp import custom_bwd, custom_fwd

# Define the custom CUDA kernel for fused 3D Convolution and ReLU
conv3d_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_relu_forward_kernel(const scalar_t* __restrict__ input,
                                          const scalar_t* __restrict__ weight,
                                          scalar_t* __restrict__ output,
                                          int batch_size,
                                          int in_channels,
                                          int out_channels,
                                          int in_depth, int in_height, int in_width,
                                          int kernel_size,
                                          int out_depth, int out_height, int out_width,
                                          int padding,
                                          int stride) {

    CUDA_KERNEL_LOOP(output_index, batch_size * out_channels * out_depth * out_height * out_width) {
        int w_out = output_index % out_width;
        int h_out = (output_index / out_width) % out_height;
        int d_out = (output_index / (out_width * out_height)) % out_depth;
        int channel_out = (output_index / (out_width * out_height * out_depth)) % out_channels;
        int n = output_index / (out_channels * out_depth * out_height * out_width);

        scalar_t sum = 0;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int d_in = d_out * stride - padding + kd;
                        int h_in = h_out * stride - padding + kh;
                        int w_in = w_out * stride - padding + kw;

                        if (d_in >= 0 && d_in < in_depth &&
                            h_in >= 0 && h_in < in_height &&
                            w_in >= 0 && w_in < in_width) {
                            sum += input[n * in_channels * in_depth * in_height * in_width +
                                        c_in * in_depth * in_height * in_width +
                                        d_in * in_height * in_width +
                                        h_in * in_width + w_in] *
                                   weight[channel_out * in_channels * kernel_size * kernel_size * kernel_size +
                                          c_in * kernel_size * kernel_size * kernel_size +
                                          kd * kernel_size * kernel_size +
                                          kh * kernel_size +
                                          kw];
                        }
                    }
                }
            }
        }
        output[output_index] = fmaxf(sum, 0); // Apply ReLU
    }
}

torch::Tensor conv3d_relu_forward(torch::Tensor input, torch::Tensor weight,
                                 int kernel_size, int padding, int stride) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = weight.size(0);
    const int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width},
                              input.options());

    int blocks = (batch_size * out_channels * out_depth * out_height * out_width + 512 - 1) / 512;
    conv3d_relu_forward_kernel<scalar_t>
        <<<blocks, 512, 0, c10::cuda::get_current_stream()>>>(
            input.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_size, out_depth, out_height, out_width,
            padding, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_relu_forward", &conv3d_relu_forward, "Fused Conv3d + ReLU forward");
}
"""

# Compile the CUDA extension
conv3d_relu = load(name="conv3d_relu", sources=[conv3d_relu_source], verbose=False)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = False):
        super().__init__()
        # Note: Bias is omitted here as it's not included in the current kernel (can be added later if needed)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels,
                                              kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Initialize weights

        # Following parameters are placeholders for the kernel; can adjust logic if needed
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # Currently, dilation and groups are not implemented in the custom kernel; 
        # would need to add support if necessary for the problem
        assert dilation == 1, "Dilation > 1 not yet supported"
        assert groups == 1, "Groups > 1 not yet supported"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        # Disable AMP to avoid precision mismatch
        x = x.float()
        weight = self.weight.float()
        return conv3d_relu.conv3d_relu_forward(x, weight, self.kernel_size[0], self.padding, self.stride)