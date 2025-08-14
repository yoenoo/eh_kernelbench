cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool3d
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void maxpool3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int64_t batch_size, int64_t channels,
    int64_t input_dim1, int64_t input_dim2, int64_t input_dim3,
    int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation,
    int64_t output_dim1, int64_t output_dim2, int64_t output_dim3) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_dim1 * output_dim2 * output_dim3) {
        return;
    }

    int64_t b = idx / (channels * output_dim1 * output_dim2 * output_dim3);
    int64_t c = (idx / (output_dim1 * output_dim2 * output_dim3)) % channels;
    int64_t out_d = (idx / (output_dim2 * output_dim3)) % output_dim1;
    int64_t out_h = (idx / output_dim3) % output_dim2;
    int64_t out_w = idx % output_dim3;

    scalar_t max_val = -FLT_MAX;
    int64_t max_idx = 0;

    for (int64_t k_d = 0; k_d < kernel_size; ++k_d) {
        for (int64_t k_h = 0; k_h < kernel_size; ++k_h) {
            for (int64_t k_w = 0; k_w < kernel_size; ++k_w) {
                int64_t d = out_d * stride + k_d * dilation - padding;
                int64_t h = out_h * stride + k_h * dilation - padding;
                int64_t w = out_w * stride + k_w * dilation - padding;

                if (d < 0 || d >= input_dim1 || h < 0 || h >= input_dim2 || w < 0 || w >= input_dim3) {
                    continue;
                }

                scalar_t val = input[b][c][d][h][w];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    output[b][c][out_d][out_h][out_w] = max_val;
}

std::vector<torch::Tensor> maxpool3d_forward_cuda(
    torch::Tensor input,
    int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation,
    int64_t ceil_mode) {

    auto output_size = at::max_poolNd_shape(
        input.sizes(),
        {kernel_size, kernel_size, kernel_size},
        {stride, stride, stride},
        {padding, padding, padding},
        {dilation, dilation, dilation},
        ceil_mode);

    auto output = torch::empty(output_size, input.options());

    const int64_t batch_size = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t input_dim1 = input.size(2);
    const int64_t input_dim2 = input.size(3);
    const int64_t input_dim3 = input.size(4);
    const int64_t output_dim1 = output.size(2);
    const int64_t output_dim2 = output.size(3);
    const int64_t output_dim3 = output.size(4);

    const int block_size = 256;
    const int num_elements = batch_size * channels * output_dim1 * output_dim2 * output_dim3;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool3d_forward_cuda", ([&] {
        auto input_acc = input.packed_accessor<Scalar,5>();
        auto output_acc = output.packed_accessor<Scalar,5>();

        maxpool3d_forward_kernel<Scalar><<<grid_size, block_size>>>(
            input_acc,
            output_acc,
            batch_size, channels,
            input_dim1, input_dim2, input_dim3,
            kernel_size, stride,
            padding, dilation,
            output_dim1, output_dim2, output_dim3);
    }));

    return {output};
}
"""

maxpool3d_cpp_source = R"(
std::vector<torch::Tensor> maxpool3d_forward_cuda(
    torch::Tensor input,
    int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation,
    int64_t ceil_mode);
)"

# Compile the inline CUDA code for MaxPool3d
maxpool3d_cuda = load_inline(
    name="maxpool3d_cuda",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices  # Currently not supported in custom kernel
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_indices:
            raise NotImplementedError("Custom kernel does not support return_indices yet.")
        # The custom kernel only returns output tensor
        outputs = maxpool3d_cuda.maxpool3d_forward_cuda(
            x.cuda(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode
        )
        return outputs[0]