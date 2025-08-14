import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class DepthwiseConv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride=1, padding=0):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        batch_size, in_channels, in_height, in_width = input.shape
        kernel_size = weight.size(2)
        out_height = (in_height + 2 * padding - kernel_size) // stride + 1
        out_width = (in_width + 2 * padding - kernel_size) // stride + 1
        
        output = torch.zeros(batch_size, in_channels, out_height, out_width, device=input.device)
        
        n = output.numel()
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        
        depthwise_conv_forward_kernel = load_inline(
            name="depthwise_conv_forward",
            cuda_sources=f"""
                #include <torch/extension.h>
                template <typename scalar_t>
                __global__ void depthwise_conv2d_kernel(
                    const scalar_t* __restrict__ input,
                    const scalar_t* __restrict__ weight,
                    scalar_t* __restrict__ output,
                    const int batch_size,
                    const int in_channels,
                    const int in_height,
                    const int in_width,
                    const int kernel_size,
                    const int out_height,
                    const int out_width,
                    const int stride,
                    const int padding
                ) {{
                    const int output_size = out_height * out_width;
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    
                    if (idx >= n) return;
                    
                    const int w = idx % out_width;
                    const int h = (idx / out_width) % out_height;
                    const int c = (idx / (out_width * out_height)) % in_channels;
                    const int n = idx / (out_channels * out_height * out_width);
                    
                    const int in_offset = n * in_channels * in_height * in_width + c * in_height * in_width;
                    const int out_offset = n * in_channels * out_height * out_width + c * out_height * out_width;
                    
                    scalar_t sum = 0.0;
                    for (int ki = 0; ki < kernel_size; ++ki) {{
                        for (int kj = 0; kj < kernel_size; ++kj) {{
                            const int hi = h * stride + ki - padding;
                            const int wi = w * stride + kj - padding;
                            
                            if (hi >=0 && hi < in_height && wi >=0 && wi < in_width) {{
                                sum += input[in_offset + hi * in_width + wi] * 
                                       weight[c * kernel_size*kernel_size + ki * kernel_size + kj];
                            }}
                        }}
                    }}
                    output[out_offset + h * out_width + w] = sum;
                }}
                
                at::Tensor depthwise_conv_forward(
                    at::Tensor input,
                    at::Tensor weight,
                    int stride,
                    int padding
                ) {{
                    const auto batch_size = input.size(0);
                    const auto in_channels = input.size(1);
                    const auto in_height = input.size(2);
                    const auto in_width = input.size(3);
                    const auto kernel_size = weight.size(2);
                    const auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
                    const auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
                    
                    auto output = at::empty({{batch_size, in_channels, out_height, out_width}}, 
                        input.options());
                    
                    const int n = output.numel();
                    const dim3 block(256);
                    const dim3 grid((n + block.x - 1) / block.x);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv_forward", ([&] {{
                        depthwise_conv2d_kernel<scalar_t><<<grid, block>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            output.data<scalar_t>(),
                            batch_size, in_channels, in_height, in_width,
                            kernel_size, out_height, out_width,
                            stride, padding
                        );
                    }}));
                    
                    return output;
                }}
            """,
            functions=["depthwise_conv_forward"],
            verbose=True
        )
        return depthwise_conv_forward_kernel.depthwise_conv_forward(input, weight, stride, padding)

    @staticmethod
    def backward(ctx, grad_output):
        # Implementing the backward pass here would be complex and is omitted for brevity
        # In practice, this would involve another kernel for gradient computation with respect to input and weights
        raise NotImplementedError("Backward pass not implemented for the example")

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = DepthwiseConv2D.apply(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output