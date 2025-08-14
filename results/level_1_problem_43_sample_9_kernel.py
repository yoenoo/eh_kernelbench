import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool3d
max_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor<int64_t, 5, torch::RestrictPtrTraits> indices,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int batch_size,
    const int channels,
    const int input_dim1,
    const int input_dim2,
    const int input_dim3,
    const int output_dim1,
    const int output_dim2,
    const int output_dim3
) {
    const int output_size = output_dim1 * output_dim2 * output_dim3;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * channels) {
        return;
    }

    const int b = idx / channels;
    const int c = idx % channels;

    for (int d = 0; d < output_dim1; d++) {
        for (int h = 0; h < output_dim2; h++) {
            for (int w = 0; w < output_dim3; w++) {
                const int input_d_start = d * stride - padding;
                const int input_h_start = h * stride - padding;
                const int input_w_start = w * stride - padding;

                scalar_t max_val = -FLT_MAX;
                int64_t max_idx = 0;

                for (int kd = 0; kd < kernel_size; ++kd) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            const int id = input_d_start + kd * dilation;
                            const int ih = input_h_start + kh * dilation;
                            const int iw = input_w_start + kw * dilation;

                            if (id >= 0 && id < input_dim1 &&
                                ih >= 0 && ih < input_dim2 &&
                                iw >= 0 && iw < input_dim3) {
                                const scalar_t val = input[b][c][id][ih][iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = id * input_dim2 * input_dim3 + ih * input_dim3 + iw;
                                }
                            }
                        }
                    }
                }
                output[b][c][d][h][w] = max_val;
                indices[b][c][d][h][w] = max_idx;
            }
        }
    }
}

torch::Tensor max_pool3d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_dim1 = input.size(2);
    const int input_dim2 = input.size(3);
    const int input_dim3 = input.size(4);

    int output_dim1 = (input_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_dim2 = (input_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_dim3 = (input_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    if (ceil_mode) {
        output_dim1 += (input_dim1 + 2 * padding - dilation * (kernel_size - 1)) % stride != 0 ? 1 : 0;
        output_dim2 += (input_dim2 + 2 * padding - dilation * (kernel_size - 1)) % stride != 0 ? 1 : 0;
        output_dim3 += (input_dim3 + 2 * padding - dilation * (kernel_size - 1)) % stride != 0 ? 1 : 0;
    }

    torch::Tensor output = torch::empty({batch_size, channels, output_dim1, output_dim2, output_dim3}, input.options());
    torch::Tensor indices = torch::empty({batch_size, channels, output_dim1, output_dim2, output_dim3}, torch::kLong);

    const int threads = 256;
    const int blocks = (batch_size * channels + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_forward_cuda", ([&] {
        auto input_acc = input.packed_accessor<scalar_t, 5>();
        auto output_acc = output.packed_accessor<scalar_t, 5>();
        auto indices_acc = indices.packed_accessor<int64_t, 5>();

        max_pool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input_acc,
            output_acc,
            indices_acc,
            kernel_size,
            stride,
            padding,
            dilation,
            batch_size,
            channels,
            input_dim1,
            input_dim2,
            input_dim3,
            output_dim1,
            output_dim2,
            output_dim3
        );
    }));

    return return_indices ? std::make_tuple(output, indices) : output;
}
"""

max_pool3d_cpp_source = (
    "std::tuple<torch::Tensor, torch::Tensor> max_pool3d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation, bool return_indices, bool ceil_mode);"
)

max_pool3d = load_inline(
    name="max_pool3d",
    cpp_sources=max_pool3d_cpp_source,
    cuda_sources=max_pool3d_source,
    functions=["max_pool3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.max_pool3d = max_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.max_pool3d.max_pool3d_forward_cuda(
            x.cuda(), 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation,
            self.return_indices,
            self.ceil_mode
        )
        return output if self.return_indices else output[0]