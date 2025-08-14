import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void ConvTranspose3DKernel(const scalar_t* input,
                                     const scalar_t* weight,
                                     scalar_t* output,
                                     const int batch_size,
                                     const int in_channels,
                                     const int out_channels,
                                     const int kT, const int kH, const int kW,
                                     const int dT, const int dH, const int dW,
                                     const int pT, const int pH, const int pW,
                                     const int o_pT, const int o_pH, const int o_pW,
                                     const int itD, const int itH, const int itW,
                                     const int otD, const int otH, const int otW) {
    const int out_depth = otD + o_pT * 2;
    const int out_height = otH + o_pH * 2;
    const int out_width = otW + o_pW * 2;

    const int in_depth_idx = otD * dT - pT;
    const int in_height_idx = otH * dH - pH;
    const int in_width_idx = otW * dW - pW;

    const int output_idx = (
        blockIdx.x * (out_depth * out_height * out_width) +
        otD * out_height * out_width +
        otH * out_width +
        otW
    );

    scalar_t val = 0;
    for (int k_depth = 0; k_depth < kT; ++k_depth) {
        const int in_depth = in_depth_idx + k_depth;
        if (in_depth < 0 || in_depth >= itD) continue;

        for (int k_height = 0; k_height < kH; ++k_height) {
            const int in_height = in_height_idx + k_height;
            if (in_height < 0 || in_height >= itH) continue;

            for (int k_width = 0; k_width < kW; ++k_width) {
                const int in_width = in_width_idx + k_width;
                if (in_width < 0 || in_width >= itW) continue;

                for (int in_c = 0; in_c < in_channels; ++in_c) {
                    for (int out_c = 0; out_c < out_channels; ++out_c) {
                        const int w_offset = (
                            out_c * in_channels * kT * kH * kW +
                            in_c * kT * kH * kW +
                            k_depth * kH * kW +
                            k_height * kW +
                            k_width
                        );
                        const int in_offset = (
                            blockIdx.x * in_channels * itD * itH * itW +
                            in_c * itD * itH * itW +
                            in_depth * itH * itW +
                            in_height * itW +
                            in_width
                        );
                        val += weight[w_offset] * input[in_offset];
                    }
                }
            }
        }
    }
    output[output_idx] = val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int strideT, int strideH, int strideW,
    int padT, int padH, int padW,
    int output_padT, int output_padH, int output_padW) 
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int itD = input.size(2);
    const int itH = input.size(3);
    const int itW = input.size(4);

    const int out_channels = weight.size(0);
    const int kT = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int otD = (itD - 1) * strideT - 2 * padT + kT + output_padT;
    const int otH = (itH - 1) * strideH - 2 * padH + kH + output_padH;
    const int otW = (itW - 1) * strideW - 2 * padW + kW + output_padW;

    auto output_options = torch::TensorOptions()
        .like(input)
        .dtype(input.dtype());

    torch::Tensor output = torch::zeros({batch_size, out_channels, otD, otH, otW}, output_options);

    dim3 threads(256);
    dim3 blocks(
        batch_size * out_channels,
        1,
        1
    );

    const int shared_mem = 0;
    const int kT_dim = weight.size(2);
    const int kH_dim = weight.size(3);
    const int kW_dim = weight.size(4);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "conv_transpose3d_cuda", ([&] {
        ConvTranspose3DKernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kT_dim, kH_dim, kW_dim,
            strideT, strideH, strideW,
            padT, padH, padW,
            output_padT, output_padH, output_padW,
            itD, itH, itW,
            otD, otH, otW);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int strideT, int strideH, int strideW, int padT, int padH, int padW, int output_padT, int output_padH, int output_padW);"
)

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)
        self.groups = groups

        # Initialize weight like PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            *self.kernel_size
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2]
        )