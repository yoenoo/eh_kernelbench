import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

// Include the THNN/THCUNN headers for the original implementation details
#include <THCUNN/THCUNN.h>
#include <THC/THC.h>

// Forward declaration of the CUDA kernel
void THNN_convTranspose3DUpdateOutput_cuda(
    at::Tensor input,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor weight,
    at::Tensor bias,
    int64_t kW, int64_t kD, int64_t kH,
    int64_t dW, int64_t dD, int64_t dH,
    int64_t padW, int64_t padD, int64_t padH,
    int64_t adjW, int64_t adjD, int64_t adjH,
    int64_t groups,
    bool scale_shift,
    at::Tensor scale,
    at::Tensor shift,
    bool verify_loading,
    bool is_result
);

// Wrap the existing THCUNN function for PyTorch compatibility
torch::Tensor my_conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t strideD, int64_t strideH, int64_t strideW,
    int64_t paddingD, int64_t paddingH, int64_t paddingW,
    int64_t output_paddingD, int64_t output_paddingH, int64_t output_paddingW,
    int64_t groups
) {
    // The original function parameters follow THCUNN's ordering and semantics
    // Note: This code assumes the same calculation of output size as PyTorch's ConvTranspose3d
    // Compute the input and output dimensions
    auto input_size = input.sizes();
    int64_t batch_size = input_size[0];
    int64_t in_channels = input_size[1];
    int64_t input_depth = input_size[2];
    int64_t input_height = input_size[3];
    int64_t input_width = input_size[4];

    // Calculate output dimensions
    int64_t out_depth = (input_depth - 1) * strideD - 2 * paddingD + kernelD + output_paddingD;
    int64_t out_height = (input_height - 1) * strideH - 2 * paddingH + kernelH + output_paddingH;
    int64_t out_width = (input_width - 1) * strideW - 2 * paddingW + kernelW + output_paddingW;

    // Initialize the output tensor
    auto output = at::empty({batch_size, weight.size(1), out_depth, out_height, out_width}, input.options());

    // Extract parameters from the weight tensor
    int64_t kernelD = weight.size(2);
    int64_t kernelH = weight.size(3);
    int64_t kernelW = weight.size(4);

    // Convert strides into the order expected by the kernel
    // (THNN uses (tY, xY, zY), which corresponds to (depth, height, width) here)
    int64_t dD = strideD, dH = strideH, dW = strideW;

    // Convert padding similarly
    int64_t padD = paddingD, padH = paddingH, padW = paddingW;

    // Output padding is mapped to adj (adjustment) in THNN's terminology
    int64_t adjD = output_paddingD, adjH = output_paddingH, adjW = output_paddingW;

    // The existing bias tensor may be empty (if bias=False)
    auto bias_ = (bias.defined()) ? bias : at::empty({0}, input.options());

    // Call the wrapped THNN function
    THCState* state = at::cuda::currentCUDAContext()->getTHCState();
    THNN_convTranspose3DUpdateOutput_cuda(
        input,
        at::Tensor(),
        output,
        weight,
        bias_,
        kernelW, kernelD, kernelH,
        dW, dD, dH,
        padW, padD, padH,
        adjW, adjD, adjH,
        groups,
        false,
        at::Tensor(), at::Tensor(),
        false, true
    );

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

// Function declarations for PyTorch extension
torch::Tensor my_conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t strideD, int64_t strideH, int64_t strideW,
    int64_t paddingD, int64_t paddingH, int64_t paddingW,
    int64_t output_paddingD, int64_t output_paddingH, int64_t output_paddingW,
    int64_t groups
);
"""

# Compile the custom ConvTranspose3D operator (reusing existing implementation)
custom_conv_transpose3d = load_inline(
    name="custom_conv_transpose3d",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=[
        "torch::Tensor my_conv_transpose3d_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)"
    ],
    verbose=False,
    with_cuda=True,
    extra_ldflags=["-lthnn"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to ConvTranspose3d
        kernel_size_depth, kernel_size_height, kernel_size_width = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size_depth, kernel_size_height, kernel_size_width))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize using the same method as PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Get parameters needed for the custom CUDA kernel
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        adj_d, adj_h, adj_w = self.output_padding

        # Pass tensors and parameters to the custom kernel
        return custom_conv_transpose3d.my_conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            adj_d, adj_h, adj_w,
            self.groups
        )