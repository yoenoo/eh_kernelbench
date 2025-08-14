cuda
import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

def get_elementwise_add():
    elementwise_add_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            out[idx] = a[idx] + b[idx];
        }
    }

    torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
        auto size = a.numel();
        auto out = torch::zeros_like(a);

        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;

        elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

        return out;
    }
    """

    elementwise_add_cpp_source = (
        "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
    )

    elementwise_add = load_inline(
        name="elementwise_add",
        cpp_sources=elementwise_add_cpp_source,
        cuda_sources=elementwise_add_source,
        functions=["elementwise_add_cuda"],
        verbose=True,
    )

    return elementwise_add

def get_conv_transpose_1d_custom(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias):
    conv_transpose_1d_source = f"""
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void conv_transpose_1d_kernel(
        const float* input,
        const float* weight,
        float* output,
        int batch_size,
        int in_channels,
        int out_channels,
        int kernel_size,
        int input_length,
        int output_length,
        int stride,
        int padding,
        int output_padding,
        int groups
    ) {{
        // Implement custom conv transpose 1D logic here...
    }}

    torch::Tensor conv_transpose_1d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        int stride,
        int padding,
        int output_padding,
        int groups
    ) {{
        // Logic to compute output dimensions and call kernel
        return output;
    }}
    """

    conv_transpose_1d_cpp_source = (
        "torch::Tensor conv_transpose_1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups);"
    )

    options = {
        "name": "conv_transpose_1d_custom",
        "cpp_sources": conv_transpose_1d_cpp_source,
        "cuda_sources": conv_transpose_1d_source,
        "functions": ["conv_transpose_1d_cuda"],
        "verbose": True
    }

    return load_inline(**options)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Create parameters as in PyTorch's ConvTranspose1d
        weight_size = (in_channels, out_channels // groups, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(*weight_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Compile custom kernels
        self.conv_transpose_1d = get_conv_transpose_1d_custom(
            in_channels, out_channels, kernel_size,
            stride, padding, output_padding, groups, bias
        )

    def forward(self, x):
        output = self.conv_transpose_1d.conv_transpose_1d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        if self.bias is not None:
            # Use custom element-wise add for bias
            elementwise_add = get_elementwise_add()
            output = elementwise_add.elementwise_add_cuda(output, self.bias.view(1, -1, 1))
        return output