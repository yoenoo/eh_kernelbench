import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def depthwise_conv2d_kernel(N, C, H, W, K, S, P, x, weight, out, bias):
    # Custom CUDA kernel for depthwise convolution
    kernel_code = f"""
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

    template <typename scalar_t>
    __global__ void depthwise_conv2d_kernel(
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ weight,
        scalar_t* __restrict__ out,
        const int N, const int C, const int H, const int W,
        const int K, const int S, const int P,
        const scalar_t* __restrict__ bias)
    {{
        CUDA_KERNEL_LOOP(output_index, N * C * (H - K + 2 * P + 1)/S * (W - K + 2 * P + 1)/S) {{
            int c = output_index / ( ((H - K + 2 * P)/S + 1) * ((W - K + 2 * P)/S + 1) );
            int oh = (output_index / ((W - K + 2 * P)/S + 1)) % ((H - K + 2 * P)/S + 1);
            int ow = output_index % ((W - K + 2 * P)/S + 1);

            oh = oh * S - P;
            ow = ow * S - P;

            scalar_t val = 0;
            for (int kh = 0; kh < K; ++kh) {{
                for (int kw = 0; kw < K; ++kw) {{
                    int h = oh + kh;
                    int w = ow + kw;
                    if (h >= 0 && h < H && w >= 0 && w < W) {{
                        val += x[c + C * (h + P + kh/S * S)] * weight[kh * K + kw];
                    }}
                }}
            }}
            out[output_index] = val + (bias ? bias[c] : 0);
        }}
    }}

    torch::Tensor depthwise_conv2d_forward(
        torch::Tensor x,
        torch::Tensor weight,
        torch::Tensor bias,
        int kernel_size,
        int stride,
        int padding) {{
        const int N = x.size(0);
        const int C = x.size(1);
        const int H = x.size(2);
        const int W = x.size(3);
        const int K = kernel_size;
        const int S = stride;
        const int P = padding;

        auto padded_H = H + 2 * P;
        auto padded_W = W + 2 * P;
        auto out_H = (padded_H - K) / S + 1;
        auto out_W = (padded_W - K) / S + 1;

        auto out = torch::empty({{N, C, out_H, out_W}}, x.options());

        dim3 blocks(N * C * out_H * out_W / 256 + 1);
        dim3 threads(256);

        auto stream = c10::cuda::getCurrentCUDAStream();
        depthwise_conv2d_kernel<float><<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, H, W,
            K, S, P,
            bias.data_ptr<float>());

        return out;
    }}
    """
    module = load_inline(
        name="depthwise_conv2d",
        cpp_sources="",
        cuda_sources=kernel_code,
        functions=["depthwise_conv2d_forward"],
        verbose=False
    )
    return module.depthwise_conv2d_forward(x, weight, bias, K, S, P)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size*kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return depthwise_conv2d_kernel(
            x.size(0),
            self.in_channels,
            x.size(2),
            x.size(3),
            self.kernel_size,
            self.stride,
            self.padding,
            x,
            self.weight,
            self.bias
        )