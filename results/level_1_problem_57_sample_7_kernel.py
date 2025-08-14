import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        # Initialize the standard ConvTranspose2d to get weights and bias
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
        
        # Inline CUDA kernel for the custom implementation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Load the custom CUDA kernel
        self.conv_t = load_inline(
            name="conv_transpose2d",
            cuda_sources="""
                #include <torch/extension.h>
                #include <ATen/cuda/CUDAContext.h>

                __global__ void conv_transpose2d_kernel(
                    const float* input, const float* weight, float* output,
                    int batch_size, int in_channels, int out_channels,
                    int kernel_size, int input_height, int input_width,
                    int output_height, int output_width,
                    int stride, int padding, int output_padding, int groups) {

                    // Index calculations for output tensor
                    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
                    int b_idx = blockIdx.z;

                    if (w_idx >= output_width || h_idx >= output_height || b_idx >= batch_size)
                        return;

                    // Compute the output's channel index and group
                    int out_ch = blockIdx.z % out_channels;  // Simplified for illustration
                    int group_id = out_ch / (out_channels / groups);

                    float val = 0.0;
                    for (int k_h = 0; k_h < kernel_size; ++k_h) {
                        for (int k_w = 0; k_w < kernel_size; ++k_w) {
                            // Determine input positions
                            int input_h = (h_idx + padding - k_h) / stride;
                            int input_w = (w_idx + padding - k_w) / stride;

                            // Check bounds
                            if (input_h < 0 || input_h >= input_height || input_w < 0 || input_w >= input_width)
                                continue;

                            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                                // Weights are [out_ch, in_ch, kh, kw] for conv, so transpose might have different arrangement
                                // Assuming weights are [in_channels, out_channels/groups, kernel_size, kernel_size]
                                int w_offset = in_ch * (out_channels/groups) * kernel_size * kernel_size
                                             + out_ch * kernel_size * kernel_size 
                                             + k_h * kernel_size + k_w;
                                val += input[b_idx * in_channels * input_height * input_width 
                                            + in_ch * input_height * input_width
                                            + input_h * input_width + input_w] 
                                        * weight[w_offset];
                            }
                        }
                    }
                    output[b_idx * out_channels * output_height * output_width
                          + out_ch * output_height * output_width
                          + h_idx * output_width + w_idx] = val;
                }

                torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                                    int batch_size, int in_channels, int out_channels,
                                                    int kernel_size, int input_height, int input_width,
                                                    int output_height, int output_width,
                                                    int stride, int padding, int output_padding, int groups) {

                    dim3 threads(16, 16);  // Thread block size
                    dim3 blocks(
                        (output_width + threads.x - 1) / threads.x,
                        (output_height + threads.y - 1) / threads.y,
                        batch_size  // Batch processing in z-dimension
                    );

                    AT_ASSERT(input.device().type() == torch::kCUDA);
                    AT_ASSERT(weight.device().type() == torch::kCUDA);

                    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

                    conv_transpose2d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
                        batch_size, in_channels, out_channels,
                        kernel_size, input_height, input_width,
                        output_height, output_width,
                        stride, padding, output_padding, groups
                    );
                    return output;
                }
            """,
            functions=["conv_transpose2d_cuda"],
            verbose=True
        )

    def forward(self, x):
        # Get the parameters from the existing conv_transpose
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else None

        batch_size, in_channels, input_h, input_w = x.shape
        # Compute output dimensions according to transpose conv formula
        output_h = (input_h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_w = (input_w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Call the custom CUDA kernel
        output = self.conv_t.conv_transpose2d_cuda(
            x, weight,
            batch_size, in_channels, weight.size(0),  # out_channels is weight's first dim?
            self.kernel_size, input_h, input_w,
            output_h, output_w,
            self.stride, self.padding, self.output_padding, self.groups
        )

        if bias is not None:
            output += bias.view(1, -1, 1, 1)  # Applying bias if present

        return output

# The original get_inputs and get_init_inputs remain unchanged