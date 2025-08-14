import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Transposed 1D Convolution
transposed_conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;\
       i += blockDim.x * gridDim.x)

at::Tensor transposed_conv1d_forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::optional<at::Tensor> &bias_opt,
    int stride,
    int padding,
    int dilation,
    bool is_transposed) {

    const auto bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int effective_kernel_size = dilation * (kernel_size - 1) + 1;
    const int out_length = (in_length - 1) * stride - 2 * padding + effective_kernel_size;

    auto output = at::empty({batch_size, out_channels, out_length}, input.options());

    const int blocks = (batch_size * out_channels * out_length + 256 - 1) / 256;
    dim3 grid(blocks);
    dim3 block(256);

    // Define the CUDA kernel for forward pass of transposed conv
    AT_DISPATCH_FLOATING_TYPES(input.type(), "transposed_conv1d_forward", ([&] {
        auto input_data = input.data<scalar_t>();
        auto weight_data = weight.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        auto bias_data = bias.data<scalar_t>();

        cudaMemset(output_data, 0, output.numel() * sizeof(scalar_t));

        transposed_conv_forward_kernel<<<grid, block>>>(
            input_data, weight_data, output_data, bias_data,
            batch_size, in_channels, out_channels, in_length, out_length, kernel_size,
            stride, padding, dilation, is_transposed, bias.defined());

        cudaDeviceSynchronize();
    }));

    return output;
}

// Define the kernel function
template <typename scalar_t>
__global__ void transposed_conv_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_length,
    const int out_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const bool is_transposed,
    const bool has_bias) {

    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * out_length) {
        int batch_idx = output_index / (out_channels * out_length);
        int out_channel_idx = (output_index / out_length) % out_channels;
        int out_pos = output_index % out_length;

        scalar_t val = 0;

        // Iterate over input channels
        for (int in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx) {
            // Iterate over kernel elements
            for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
                int dilated_k = kernel_idx * dilation;
                int input_pos = out_pos - (dilated_k - padding);

                // Check if input position is valid
                if (input_pos >= 0 && input_pos < in_length) {
                    int weight_offset = (out_channel_idx * in_channels + in_channel_idx) * kernel_size + kernel_idx;
                    val += input[batch_idx * in_channels * in_length + in_channel_idx * in_length + input_pos] *
                           weight[weight_offset];
                }
            }
        }

        if (has_bias) {
            val += bias[out_channel_idx];
        }

        output[output_index] = val;
    }
}

// Define the C++ wrapper function for PyTorch
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

torch::Tensor transposed_conv1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    bool is_transposed) {

    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias.has_value()) CHECK_CUDA(bias.value());

    return transposed_conv1d_forward(input, weight, bias, stride, padding, dilation, is_transposed);
}

"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor transposed_conv1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    bool is_transposed);
"""

# Compile the custom CUDA kernel
transposed_conv1d = load_inline(
    name="transposed_conv1d",
    cpp_sources=cpp_source,
    cuda_sources=transposed_conv1d_source,
    functions=["transposed_conv1d_forward_cuda"],
    verbose=True,
    with_cuda=True,
    extra_cflags=['-std=c++14'],
    extra_cuda_cflags=['-std=c++14', '-gencode=arch=compute_70,code=sm_70']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize parameters with same method as PyTorch's ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Get bias as optional tensor
        bias_opt = self.bias if self.bias is not None else torch.tensor(None)
        
        # Call custom CUDA kernel
        return transposed_conv1d.transposed_conv1d_forward_cuda(
            x, self.weight, bias_opt, self.stride, self.padding, self.dilation, True)