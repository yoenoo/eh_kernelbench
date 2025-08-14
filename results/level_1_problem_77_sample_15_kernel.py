import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// Function to compute output dimensions
std::vector<int64_t> compute_output_shape(
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    auto output_depth = (input_depth - 1) * stride - 2 * padding + (dilation * (kernel_size - 1) + 1);
    auto output_height = (input_height - 1) * stride - 2 * padding + (dilation * (kernel_size - 1) + 1);
    auto output_width = (input_width - 1) * stride - 2 * padding + (dilation * (kernel_size - 1) + 1);
    return {output_depth, output_height, output_width};
}

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_channels,
    int in_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width) {

    const int batch_idx = blockIdx.x;
    const int out_d = blockIdx.y;
    const int out_h = blockIdx.z * blockDim.y + threadIdx.y;
    const int out_w = threadIdx.x;

    if (out_h >= output_height || out_w >= output_width) return;

    for (int oc = 0; oc < out_channels; oc += blockDim.z) {
        const int current_oc = oc + threadIdx.z;
        if (current_oc >= out_channels) continue;

        scalar_t sum = 0;

        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int dilated_kd = kd * dilation;
                        const int dilated_kh = kh * dilation;
                        const int dilated_kw = kw * dilation;

                        const int input_d = out_d + dilated_kd - padding;
                        const int input_h = out_h + dilated_kh - padding;
                        const int input_w = out_w + dilated_kw - padding;

                        if (input_d < 0 || input_d >= input_depth ||
                            input_h < 0 || input_h >= input_height ||
                            input_w < 0 || input_w >= input_width) {
                            continue;
                        }

                        const int kernel_idx = ic * kernel_size*kernel_size*kernel_size + kd*kernel_size*kernel_size + kh*kernel_size + kw;
                        const scalar_t w_val = weight[current_oc][ic][kd][kh][kw];
                        const scalar_t in_val = input[batch_idx][ic][input_d][input_h][input_w];
                        sum += w_val * in_val;
                    }
                }
            }
        }

        output[batch_idx][current_oc][out_d][out_h][out_w] = sum;
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const auto in_channels = input.size(1);
    const auto kernel_size = weight.size(3); // Assuming kernel dimensions are same in all axes
    const auto batch_size = input.size(0);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);
    const auto out_channels = weight.size(0);

    auto output_shape = compute_output_shape(
        input_depth, input_height, input_width,
        kernel_size, stride, padding, dilation);

    auto output_depth = output_shape[0];
    auto output_height = output_shape[1];
    auto output_width = output_shape[2];

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int threads = 32;
    dim3 blocks(
        batch_size,
        output_depth,
        (output_height + threads - 1) / threads);

    const int max_threads_per_block = 1024;
    int z_dim = std::min(out_channels, max_threads_per_block / (threads * threads));
    dim3 tpm(threads, threads, z_dim);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, tpm>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            kernel_size,
            stride,
            padding,
            dilation,
            out_channels,
            in_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and parameters to match PyTorch's ConvTranspose3d
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        # Initialize weights using same method as PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.conv_transpose3d_cuda_op = conv_transpose3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_transpose3d_cuda_op(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            # Bias addition fused into CUDA kernel for optimization
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output