cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Load the custom CUDA kernel
        self.avg_pool_cuda = load_inline(
            name="custom_avg_pool",
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <vector>
                
                template <typename scalar_t>
                __global__ void avg_pool1d_kernel(
                    const torch::PackedTensorAccessor<scalar_t,3> input,
                    torch::PackedTensorAccessor<scalar_t,3> output,
                    const int kernel_size,
                    const int stride,
                    const int padding,
                    const int batch_size,
                    const int channels,
                    const int input_length,
                    const int output_length
                ) {{
                    int batch_idx = blockIdx.x;
                    int channel_idx = blockIdx.y;
                    int out_pos = threadIdx.x;
                    
                    const int input_start = out_pos * stride - padding;
                    const int input_end = input_start + kernel_size;
                    
                    scalar_t sum = 0.0;
                    for (int i = input_start; i < input_end; ++i) {{
                        if (i >= 0 && i < input_length) {{
                            sum += input[batch_idx][channel_idx][i];
                        }}
                    }}
                    output[batch_idx][channel_idx][out_pos] = sum / static_cast<scalar_t>(kernel_size);
                }}

                torch::Tensor custom_avg_pool(
                    torch::Tensor input,
                    int kernel_size,
                    int stride,
                    int padding
                ) {{
                    const int batch_size = input.size(0);
                    const int channels = input.size(1);
                    const int input_length = input.size(2);
                    
                    const int output_length = 
                        (input_length + 2 * padding - kernel_size) / stride + 1;
                    
                    auto options = torch::TensorOptions()
                        .dtype(input.dtype())
                        .device(input.device());
                    torch::Tensor output = torch::empty({{batch_size, channels, output_length}}, options);
                    
                    const dim3 blocks(batch_size, channels, 1);
                    const dim3 threads(output_length, 1, 1);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool1d", ([&] {{
                        avg_pool1d_kernel<scalar_t><<<blocks, threads>>>(
                            input.packed_accessor<scalar_t,3>(),
                            output.packed_accessor<scalar_t,3>(),
                            kernel_size,
                            stride,
                            padding,
                            batch_size,
                            channels,
                            input_length,
                            output_length
                        );
                    }}));
                    
                    return output;
                }}
            """,
            functions=["custom_avg_pool"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.custom_avg_pool(
            x, self.kernel_size, self.stride, self.padding
        )