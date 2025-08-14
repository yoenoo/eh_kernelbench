import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Define the custom CUDA kernel for transposed 1D convolution
        transposed_conv1d_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        __global__ void transposed_conv1d_kernel(
            const float* __restrict__ input,
            const float* __restrict__ weight,
            float* __restrict__ output,
            const int batch_size,
            const int in_channels,
            const int out_channels,
            const int kernel_size,
            const int input_length,
            const int output_length,
            const int stride,
            const int padding,
            const int dilation) 
        {
            int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (output_idx >= batch_size * output_length * out_channels) return;

            int batch = output_idx / (output_length * out_channels);
            int oc = (output_idx / output_length) % out_channels;
            int out_pos = output_idx % output_length;

            float acc = 0.0;
            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int dilated_k = k * dilation;
                    int input_pos = out_pos - (dilated_k + padding);
                    if (input_pos < 0 || input_pos >= input_length) continue;
                    int kernel_idx = (oc * in_channels + ic) * kernel_size + k;
                    int input_offset = (batch * in_channels + ic) * input_length + input_pos;
                    acc += weight[kernel_idx] * input[input_offset];
                }
            }
            if (self.bias is not None) {
                acc += self.bias[oc];
            }
            int output_offset = (batch * out_channels + oc) * output_length + out_pos;
            output[output_offset] = acc;
        }

        torch::Tensor transposed_conv1d_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int batch_size,
            int in_channels,
            int out_channels,
            int kernel_size,
            int input_length,
            int output_length,
            int stride,
            int padding,
            int dilation) 
        {
            auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

            const int threads = 256;
            const int elements = batch_size * out_channels * output_length;
            const int blocks = (elements + threads - 1) / threads;

            transposed_conv1d_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                in_channels,
                out_channels,
                kernel_size,
                input_length,
                output_length,
                stride,
                padding,
                dilation
            );

            cudaDeviceSynchronize();
            return output;
        }
        """

        # Compile the kernel
        self.transposed_conv1d = load_inline(
            name="transposed_conv1d",
            cuda_sources=transposed_conv1d_source,
           functions=["transposed_conv1d_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.size(2)
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        output = self.transposed_conv1d.transposed_conv1d_cuda(
            x,
            self.weight,
            self.bias,
            x.size(0),
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            input_length,
            output_length,
            self.stride,
            self.padding,
            self.dilation
        )
        return output