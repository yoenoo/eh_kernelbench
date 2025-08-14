import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define the custom CUDA kernel for average pooling
        self.avg_pool_cuda = load_inline(
            name="avg_pool_cuda",
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>

                __global__ void avg_pool1d_kernel(const float* input, float* output,
                                                  int batch_size, int channels, int input_length,
                                                  int kernel_size, int stride, int padding) {{
                    // Calculate the output dimensions
                    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

                    // Get the thread's position in the grid
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= batch_size * channels * output_length) return;

                    // Compute the 3D coordinates (b, c, o) from the linear index
                    int o = idx % output_length;
                    int c = (idx / output_length) % channels;
                    int b = idx / (channels * output_length);

                    // Compute the start and end positions in the input
                    int start = o * stride - padding;
                    int end = start + kernel_size;

                    float sum = 0.0;
                    int valid_count = 0;

                    // Iterate over the kernel window
                    for (int i = start; i < end; ++i) {{
                        if (i >= 0 && i < input_length) {{
                            sum += input[b * channels * input_length + c * input_length + i];
                            valid_count++;
                        }}
                    }}

                    output[idx] = sum / valid_count;
                }}

                torch::Tensor avg_pool1d_forward(torch::Tensor input) {{
                    const int batch_size = input.size(0);
                    const int channels = input.size(1);
                    const int input_length = input.size(2);

                    int output_length = (input_length + 2 * {padding} - {kernel_size}) / {stride} + 1;

                    auto output = torch::empty({{batch_size, channels, output_length}}, 
                                              dtype=input.dtype(), 
                                              device=input.device());

                    const int threads_per_block = 256;
                    const int num_elements = batch_size * channels * output_length;
                    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

                    avg_pool1d_kernel<<<num_blocks, threads_per_block>>>(
                        input.data_ptr<float>(),
                        output.data_ptr<float>(),
                        batch_size, channels, input_length,
                        {kernel_size}, {stride}, {padding}
                    );

                    return output;
                }}
            """,
            functions=["avg_pool1d_forward"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly cast the inputs to float32 to ensure compatibility
        x = x.float()
        return self.avg_pool_cuda.avg_pool1d_forward(x)

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]