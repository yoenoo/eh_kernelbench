import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch import Tensor

class OptimizedConvTranspose2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, input: Tensor, weight: Tensor, bias: Tensor, 
                stride: tuple, padding: tuple, output_padding: tuple, groups: int, kernel_size: tuple,
                output_padding_int: int, input_padding: tuple):
        # Define the CUDA kernel code
        kernel_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

using Vec4 = torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits>;
using Vec = torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits>;

template <typename scalar_t>
__global__ void optimized_conv_transpose2d_kernel(
    const Vec4 input,
    const Vec4 weight,
    Vec4 output,
    int batch_size, int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups
) {
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.x + threadIdx.x;
    int out_channel = threadIdx.y * blockDim.x + threadIdx.x;

    // Add bounds checking
    if (out_channel >= out_channels || out_y >= output.size(2) || out_x >= output.size(3)) {
        return;
    }

    // Compute the input coordinates
    int in_channel_group = out_channel / (out_channels / groups);
    int group_id = out_channel % (out_channels / groups);

    int effective_kernel_h = kernel_h - 2 * padding_h;
    int effective_kernel_w = kernel_w - 2 * padding_w;

    for (int kernel_y = 0; kernel_y < kernel_h; kernel_y++) {
        for (int kernel_x = 0; kernel_x < kernel_w; kernel_x++) {
            int input_y = (out_y * stride_h) - padding_h + kernel_y;
            int input_x = (out_x * stride_w) - padding_w + kernel_x;

            if (input_y < 0 || input_y >= input.size(2) || input_x < 0 || input_x >= input.size(3)) {
                continue;
            }

            for (int in_channel = 0; in_channel < in_channels; in_channel += groups) {
                int weight_offset = (group_id * kernel_h * kernel_w + kernel_y * kernel_w + kernel_x) * in_channels / groups;
                int input_offset = batch_idx * in_channels * input.size(2) * input.size(3) +
                                    (in_channel + in_channel_group) * input.size(2) * input.size(3) +
                                    input_y * input.size(3) + input_x;
                
                atomicAdd(&output[batch_idx][out_channel][out_y][out_x],
                        weight[weight_offset + in_channel / groups] * input[input_offset]);
            }
        }
    }
}

torch::Tensor optimized_conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups,
    int kernel_h, int kernel_w
) {
    auto output_height = (input.size(2) - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    auto output_width = (input.size(3) - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({input.size(0), weight.size(0), output_height, output_width}, 
                              input.options());

    int threads = 256;
    dim3 blocks(input.size(0),
                (output_height + threads - 1) / threads,
                (output_width + threads - 1) / threads);

    dim3 threadsPerBlock(16, 16);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_conv_transpose2d_forward", ([&] {
        optimized_conv_transpose2d_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            input.size(0), input.size(1), weight.size(0),
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            groups
        );
    }));

    if (bias.defined()) {
        output += bias.view({1, -1, 1, 1});
    }

    return output;
}
"""

        # Compile the CUDA kernel
        module = load_inline(
            name="optimized_conv_transpose",
            cpp_sources="",
            cuda_sources=kernel_code,
            functions=["optimized_conv_transpose2d_forward"],
            verbose=False
        )

        # Extract parameters from the context
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        output_padding_h, output_padding_w = output_padding
        kernel_h, kernel_w = kernel_size
        input_height, input_width = input.shape[2], input.shape[3]

        # Compute output dimensions manually
        output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
        output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w

        # Run the CUDA kernel
        output = module.optimized_conv_transpose2d_forward(
            input.contiguous(), 
            weight.contiguous(), 
            bias if bias is not None else torch.empty(0),
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            groups,
            kernel_h, kernel_w
        )

        ctx.save_for_backward(input, weight, bias, 
                             torch.tensor(stride), torch.tensor(padding),
                             torch.tensor(output_padding), 
                             torch.tensor(groups), torch.tensor(kernel_size))
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass using similar kernel approach (code omitted for brevity)
        # This is a placeholder for actual implementation
        input, weight, bias, stride, padding, output_padding, groups, kernel_size = ctx.saved_tensors
        return None, None, None, None, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.kernel_size = kernel_size

        # Initialize weights similar to PyTorch's ConvTranspose2d
        weight_size = (in_channels, out_channels // groups) + kernel_size
        self.weight = nn.Parameter(torch.empty(weight_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            bias_size = (out_channels,)
            self.bias = nn.Parameter(torch.empty(bias_size))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return OptimizedConvTranspose2dFunction.apply(
            x, self.weight, self.bias if self.bias is not None else torch.empty(0),
            self.stride, self.padding, self.output_padding,
            self.groups, self.kernel_size,
            0,  # output_padding_int (not used in this simplified example)
            (0, 0)  # input_padding (not used here)
        )