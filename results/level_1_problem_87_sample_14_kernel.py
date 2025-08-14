import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.bias = bias
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

        # Custom CUDA kernel code
        self.depthwise_conv = self._compile_depthwise_conv()

    def _compile_depthwise_conv(self):
        kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void depthwise_conv_forward_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int in_channels,
            const int out_channels,
            const int height,
            const int width) {

            int batch_idx = blockIdx.x;
            int in_channel_idx = blockIdx.y;
            int out_channel_idx = threadIdx.x;

            if (out_channel_idx >= out_channels) return;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int in_offset = ((batch_idx * in_channels + in_channel_idx) * height + h) * width + w;
                    int out_offset = ((batch_idx * out_channels + out_channel_idx) * height + h) * width + w;
                    output[out_offset] += input[in_offset] * weight[out_channel_idx * in_channels + in_channel_idx];
                }
            }
        }

        torch::Tensor depthwise_conv_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
            const int batch_size = input.size(0);
            const int in_channels = input.size(1);
            const int height = input.size(2);
            const int width = input.size(3);
            const int out_channels = weight.size(0);

            auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

            dim3 blocks(batch_size, in_channels);
            dim3 threads(out_channels);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv_forward", ([&] {
                depthwise_conv_forward_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, in_channels, out_channels, height, width);
            }));

            if (bias.defined()) {
                output += bias.view(1, -1, 1, 1);
            }

            return output;
        }
        """

        kernel_cpp = "torch::Tensor depthwise_conv_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

        return load_inline(
            name="depthwise_conv",
            cpp_sources=kernel_cpp,
            cuda_sources=kernel_source,
            functions="depthwise_conv_forward",
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.depthwise_conv.depthwise_conv_forward(
            x.cuda(), self.weight.cuda(), self.bias_param.cuda() if self.bias else torch.Tensor([])
        )
        return output.cuda()