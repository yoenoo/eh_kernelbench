import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Load custom convolution kernel
        self.fast_conv = load_inline(
            name="fast_conv",
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __global__ void pointwise_conv2d_kernel(
                const torch::PackedTensorAccessor<scalar_t,4> input,
                const torch::PackedTensorAccessor<scalar_t,4> weight,
                torch::PackedTensorAccessor<scalar_t,4> output,
                const scalar_t* __restrict__ bias,
                int batch_size, int in_channels, int out_channels, int height, int width) 
            {{
                int batch_idx = blockIdx.x;
                int out_ch = threadIdx.x;
                int spatial_idx = blockIdx.y * blockDim.y + threadIdx.y;

                if (out_ch < out_channels && spatial_idx < height * width) {{
                    int h = spatial_idx / width;
                    int w = spatial_idx % width;
                    
                    scalar_t sum = 0;
                    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {{
                        sum += input[batch_idx][in_ch][h][w] * weight[out_ch][in_ch][0][0];
                    }}
                    if (bias) sum += bias[out_ch];
                    output[batch_idx][out_ch][h][w] = sum;
                }}
            }}

            torch::Tensor fast_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {{
                const int batch_size = input.size(0);
                const int in_channels = input.size(1);
                const int out_channels = weight.size(0);
                const int height = input.size(2);
                const int width = input.size(3);
                
                auto output = torch::empty({{batch_size, out_channels, height, width}}, 
                                          input.options());
                
                const int threads = 256; // out_channels + 1 for threading
                const dim3 blocks(batch_size, (height * width + threads - 1)/threads);
                const dim3 tblock(out_channels, (height * width + threads - 1)/threads);
                
                auto stream = at::cuda::getCurrentCUDAStream();
                
                AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv2d", ([&] {
                    pointwise_conv2d_kernel<scalar_t><<<blocks, tblock, 0, stream>>>(
                        input.packed_accessor<scalar_t,4>(),
                        weight.packed_accessor<scalar_t,4>(),
                        output.packed_accessor<scalar_t,4>(),
                        bias.data<scalar_t>(),
                        batch_size, in_channels, out_channels, height, width);
                }));
                
                return output;
            }}
            """,
            functions=["fast_conv2d"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fast_conv.fast_conv2d(x, self.weight, self.bias) if self.bias is not None else \
               self.fast_conv.fast_conv2d(x, self.weight, torch.empty(0, device=x.device))