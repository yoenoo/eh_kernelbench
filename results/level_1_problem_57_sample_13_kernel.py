import torch
import torch.nn as nn

def load_conv_transpose2d_cuda():
    conv_transpose2d_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <vector>

    template <typename scalar_t>
    __global__ void conv_transpose2d_kernel(
        const torch::PackedTensorAccessor<scalar_t,4> input,
        const torch::PackedTensorAccessor<scalar_t,4> weight,
        torch::PackedTensorAccessor<scalar_t,4> output,
        int out_channels,
        int in_channels,
        int kernel_size,
        int stride,
        int padding,
        int output_padding,
        int groups) {

        int b = blockIdx.x;
        int out_y = blockIdx.y;
        int out_x = blockIdx.z;

        scalar_t sum = 0;
        for (int g = 0; g < groups; ++g) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                        int in_b = b;
                        int in_ch_idx = g * (in_channels / groups) + in_ch;
                        int out_ch = g * (out_channels / groups) + ((out_y * output.stride(2) + out_x) / (kernel_size * kernel_size));
                        int in_y = out_y + kh - padding;
                        int in_x = out_x + kw - padding + output_padding;

                        if (in_y < 0 || in_y >= input.size(2) || in_x <0 || in_x >= input.size(3)) {
                            continue;
                        }
                        sum += input[in_b][in_ch_idx][in_y][in_x] * 
                            weight[out_ch][in_ch_idx][kh][kw];
                    }
                }
            }
        }
        output[b][out_ch][out_y][out_x] = sum;
    }

    torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups) {
        const int batch_size = input.size(0);
        const int out_channels = weight.size(0);
        const int in_channels = weight.size(1);
        const int kernel_size = weight.size(2);
        const int output_height = input.size(2)*stride - 2*padding + kernel_size + output_padding;
        const int output_width = input.size(3)*stride - 2*padding + kernel_size + output_padding;

        auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
        torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, output_options);

        dim3 blocks(batch_size, output_height, output_width);
        dim3 threads(256); //TODO: Tune thread count

        AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
            conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
                input.packed_accessor<scalar_t,4>(),
                weight.packed_accessor<scalar_t,4>(),
                output.packed_accessor<scalar_t,4>(),
                out_channels, in_channels, kernel_size,
                stride, padding, output_padding, groups);
        }));

        cudaDeviceSynchronize();
        return output;
    }
    """
    
    # Create a wrapper module using load_inline
    conv_transpose2d = torch.utils.cpp_extension.load_inline(
        name="conv_transpose2d",
        cpp_sources="",
        cuda_sources=conv_transpose2d_source,
        functions="conv_transpose2d_cuda",
        verbose=True
    )
    
    return conv_transpose2d

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights like PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        
        # Load the CUDA extension once during initialization
        self.conv_transpose2d_cuda = load_conv_transpose2d_cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )

# Compatibility layer for parameters
def get_init_inputs():
    return [64, 64, 3]  # Example parameters, should be replaced with actual input

def get_inputs():
    x = torch.rand(8, 64, 1024, 1024)
    return [x]