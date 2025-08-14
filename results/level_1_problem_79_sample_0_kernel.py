import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.weights = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.register_parameter('bias_param', None)

        # Define the custom CUDA kernel for transposed 1D convolution
        conv1d_transpose_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <vector>

        __global__ void conv1d_transpose_kernel(
            const float* input,
            const float* weight,
            float* output,
            const int batch_size,
            const int in_channels,
            const int out_channels,
            const int kernel_size,
            const int input_length,
            const int output_length,
            const int stride,
            const int padding,
            const int dilation,
            const bool has_bias,
            const float* bias) {
                
            extern __shared__ char sdata[];
            float* smem = reinterpret_cast<float*>(sdata);
            
            int batch_idx = blockIdx.x;
            int out_channel = blockIdx.y;
            int out_pos = threadIdx.x;

            // Calculate input position
            int eff_kernel_size = (kernel_size - 1) * dilation + 1;
            int output_padding = (output_length - 1 - (input_length - 1) * stride - eff_kernel_size + 2 * padding);
            // Output position to input position mapping
            int in_pos = (out_pos - padding - output_padding) / stride;
            in_pos = (out_pos - padding - output_padding) / stride;
            // Skip if out of input bounds
            if (out_pos >= output_length || in_pos < 0 || in_pos >= input_length) return;

            // Accumulate contributions from kernel
            float val = has_bias ? bias[out_channel] : 0;
            for (int k = 0; k < kernel_size; ++k) {
                int wpos = kernel_size - 1 - k; // Reverse kernel for transpose
                int dilated_k = wpos * dilation;
                int input_pos = out_pos - (dilated_k + padding - output_padding);
                if (input_pos >= 0 && input_pos < input_length) {
                    for (int ichan = 0; ichan < in_channels; ++ichan) {
                        val += weight[out_channel * in_channels * kernel_size + ichan * kernel_size + k] *
                            input[batch_idx * in_channels * input_length + ichan * input_length + input_pos];
                    }
                }
            }

            atomicAdd(&output[batch_idx * out_channels * output_length + out_channel * output_length + out_pos], val);
        }

        torch::Tensor conv1d_transpose_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::optional<torch::Tensor> bias,
            int stride,
            int padding,
            int dilation,
            int kernel_size) {
            
            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto input_length = input.size(2);
            const auto out_channels = weight.size(0);
            const auto output_length = (input_length - 1) * stride + ((kernel_size - 1) * dilation + 1) - 2 * padding;
            
            auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());
            
            dim3 blocks(batch_size, out_channels);
            dim3 threads(output_length);
            // Need enough shared memory for all threads
            size_t smem_size = output_length * sizeof(float);
            if (smem_size > 48 * 1024) { // Check CUDA limit
                // Handle large cases with block size reduction if needed
                // This is simplified version assuming sufficient SMEM
                smem_size = 48 * 1024;
            }
            
            conv1d_transpose_kernel<<<blocks, threads, smem_size>>>(
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
                dilation,
                bias.has_value(),
                bias.has_value() ? bias.value().data_ptr<float>() : nullptr
            );
            
            cudaDeviceSynchronize();
            return output;
        }
        """

        conv1d_transpose_cpp = "torch::Tensor conv1d_transpose_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias, int stride, int padding, int dilation, int kernel_size);"

        # Compile the custom CUDA kernel
        self.conv_transpose_op = load_inline(
            name="conv1d_transpose_cuda",
            cpp_sources=conv1d_transpose_cpp,
            cuda_sources=conv1d_transpose_source,
            functions=["conv1d_transpose_cuda"],
            verbose=True,
        )
    
    def forward(self, x):
        bias = self.bias_param if self.bias else torch.tensor([])
        return self.conv_transpose_op.conv1d_transpose_cuda(
            x,
            self.weights,
            bias if self.bias else torch.tensor([]),
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size
        )