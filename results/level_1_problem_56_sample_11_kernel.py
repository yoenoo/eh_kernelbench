import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                     const torch::PackedTensorAccessor<scalar_t,4> weight,
                                     torch::PackedTensorAccessor<scalar_t,4> output,
                                     int kernel_h, int kernel_w,
                                     int stride_h, int stride_w,
                                     int padding_h, int padding_w,
                                     int dilation_h, int dilation_w,
                                     int groups) {
    
    const int batch_idx = blockIdx.x;
    const int out_y = blockIdx.y;
    const int out_x = blockIdx.z;

    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int output_height = output.size(2);
    const int output_width = output.size(3);

    const int group_size = in_channels / groups;

    __shared__ scalar_t shared_input[32][32]; // Shared memory for input block

    scalar_t sum = 0;
    
    for (int kernel_y = 0; kernel_y < kernel_h; kernel_y++) {
        for (int kernel_x = 0; kernel_x < kernel_w; kernel_x++) {
            int in_y = out_y * stride_h - padding_h + kernel_y * dilation_h;
            int in_x = out_x * stride_w - padding_w + kernel_x * dilation_w;

            if (in_y >= 0 && in_y < input.size(2) && in_x >= 0 && in_x < input.size(3)) {
                scalar_t val = input[batch_idx][threadIdx.y][in_y][in_x];
                CUDA_KERNEL_LOOP(c, group_size) {
                    scalar_t weight_val = weight[threadIdx.y * group_size + c][kernel_y][kernel_x];
                    sum += val * weight_val;
                }
            }
        }
    }

    atomicAdd(&output[batch_idx][threadIdx.y][out_y][out_x], sum);
}

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, 
                            int kernel_h, int kernel_w, 
                            int stride_h, int stride_w, 
                            int padding_h, int padding_w, 
                            int dilation_h, int dilation_w, 
                            int groups) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Calculate output dimensions
    auto output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    auto output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(1, groups);
    dim3 blocks(batch_size, output_height, output_width);

    auto stream = at::cuda::getCurrentCUDAStream();

    conv2d_forward_kernel<float><<<blocks, threads, 0, stream>>>(
        input.packed_accessor<float,4>(),
        weight.packed_accessor<float,4>(),
        output.packed_accessor<float,4>(),
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight,
                            int kernel_h, int kernel_w,
                            int stride_h, int stride_w,
                            int padding_h, int padding_w,
                            int dilation_h, int dilation_w,
                            int groups);
"""

conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights similar to PyTorch's Conv2d
        kernel_h, kernel_w = kernel_size
        weight_shape = (out_channels, in_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            bias_shape = (out_channels,)
            self.bias = nn.Parameter(torch.empty(bias_shape))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        out = conv2d_op.conv2d_forward(
            x,
            self.weight,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups
        )
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        
        return out