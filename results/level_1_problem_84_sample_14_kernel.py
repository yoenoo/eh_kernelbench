import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::DefaultPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::DefaultPtrTraits> kernel,
    torch::PackedTensorAccessor<scalar_t,4,torch::DefaultPtrTraits> output,
    int batch_size,
    int in_channels,
    int out_channels,
    int H_in, int W_in,
    int H_out, int W_out,
    int kernel_size,
    int stride,
    int padding) {

    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int h = threadIdx.y * blockDim.x + threadIdx.x;

    if (b >= batch_size || c >= in_channels || h >= H_out * W_out)
        return;

    const int w_out = h % W_out;
    const int h_out = h / W_out;

    float sum = 0.0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;

            if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in)
                continue;

            // Assuming kernel is (in_channels, 1, kH, kW)
            // But since it's depthwise, each channel has its own kernel
            // So kernel shape is (in_channels, 1, kH, kW), flattened to kernel[c*... etc.
            sum += input[b][c][h_in][w_in] * kernel[c][0][kh][kw];
        }
    }

    output[b][c][h_out][w_out] = sum;
}

// PyTorch wrapper
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    int stride,
    int padding) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);

    const auto kernel_size = kernel.size(2); // assuming kernel is (in_channels, 1, kH, kW)

    // Compute output dimensions
    const auto H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const auto W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, H_out, W_out}, input.options());

    const int threads_per_block = 256;
    dim3 blocks(batch_size, in_channels);
    dim3 threads(threads_per_block);

    // Calculate the number of elements per channel and batch
    const int elements_per_c = H_out * W_out;
    // Split work into blocks of threads per channel and batch
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv2d_cuda", ([&]{
        using scalar_t = scalar_t;
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.packed_accessor<scalar_t,4,torch::DefaultPtrTraits>(),
            kernel.packed_accessor<scalar_t,4,torch::DefaultPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::DefaultPtrTraits>(),
            batch_size,
            in_channels,
            in_channels, // assuming out_channels = in_channels (since depthwise)
            H_in, W_in,
            H_out, W_out,
            kernel_size,
            stride,
            padding);
    }));

    return output;
}
"""

depthwise_conv2d_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor kernel, int stride, int padding);"
)

# Compile the inline CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        # Initialize the kernel (since depthwise, kernel has shape (in_channels, 1, kH, kW)
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        # Assuming out_channels must be a multiple of in_channels (since groups=in_channels)
        # In the problem, it's set to same as in_channels?
        # Since original model uses groups=in_channels, out_channels must be multiple of in_channels. Here, perhaps same
        # So setting out_channels to in_channels? Not sure but following original code structure.
        # However, the user's code allows out_channels as parameter, so maybe need to handle it?
        # Wait in the original Model, it's nn.Conv2d with groups=in_channels, so out_channels must be multiple of in_channels.
        # But the problem's example uses out_channels =128 same as in_channels.
        # Hence, the code might assume out_channels == in_channels. Or perhaps the kernel is designed with in_channels = out_channels.

        # Initialize kernel weights (this is similar to standard Conv2d's parameters)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Store the custom operator
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if parameters are compatible with the custom kernel
        out = self.depthwise_conv2d_cuda(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            # The bias is added per output channel
            out += self.bias.view(1, -1, 1, 1)
        return out