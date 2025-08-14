import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int input_length,
    const int output_length,
    const int stride,
    const int padding,
    const int dilation,
    const bool has_bias,
    const float* __restrict__ bias
) {
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    // Calculate output coordinates
    int output_pos = idx % output_length;
    int oc = (idx / output_length) % out_channels;
    int n = idx / (out_channels * output_length);

    float sum = 0;
    // Loop over kernel elements
    for (int k_idx = 0; k_idx < kernel_size; ++k_idx) {
        // Compute input position
        int d = k_idx * dilation;
        int input_pos = output_pos - d - padding;
        if (input_pos < 0 || input_pos >= input_length) continue;

        // Compute corresponding kernel weight
        int kw_idx = kernel_size - 1 - k_idx;  // Reverse kernel for transposed
        int w_offset = oc * in_channels * kernel_size + (kw_idx * in_channels);
        
        // Sum over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            sum += weight[w_offset + ic] * input[n * in_channels * input_length + ic * input_length + input_pos];
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }
    output[idx] = sum;
}

torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    // Calculate output length
    int input_length = input.size(2);
    int effective_kernel_size = dilation * (kernel_size - 1) + 1;
    int output_length = (input_length - 1) * stride - 2 * padding + effective_kernel_size;

    auto output_options = torch::TensorOptions().like(input);
    auto output = torch::empty({input.size(0), weight.size(1), output_length}, output_options);

    int threads = 256;
    int elements = output.numel();
    int blocks = (elements + threads - 1) / threads;

    // Get parameters
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    // Launch kernel
    conv1d_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_length,
        output_length,
        stride,
        padding,
        dilation,
        has_bias,
        has_bias ? bias.data_ptr<float>() : nullptr
    );

    return output;
}
"""

cpp_src = """
torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
"""

# Compile the inline CUDA extension
conv1d_transpose = load_inline(
    name="conv1d_transpose",
    cpp_sources=cpp_src,
    cuda_sources=conv1d_source,
    functions=["conv1d_transpose_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights and bias similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert parameters to correct format
        has_bias = self.bias is not None
        bias_tensor = self.bias if has_bias else torch.empty(0)
        return conv1d_transpose.conv1d_transpose_cuda(
            x.cuda(),
            self.weight.cuda(),
            bias_tensor.cuda(),
            self.stride,
            self.padding,
            self.dilation,
            has_bias
        )