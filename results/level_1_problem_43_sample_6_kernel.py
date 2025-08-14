import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        # Load custom CUDA kernel for MaxPool3D
        maxpool3d_cuda_src = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <vector>

        template <typename scalar_t>
        __global__ void maxpool3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,5> input,
                                                 torch::PackedTensorAccessor<scalar_t,5> output,
                                                 torch::PackedTensorAccessor<int,5> indices,
                                                 const int batch_size, const int channels,
                                                 const int input_dim1, const int input_dim2, const int input_dim3,
                                                 const int kernel_size, const int stride, const int padding, const int dilation,
                                                 const bool ceil_mode) {
            const int output_dim1 = (input_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            const int output_dim2 = (input_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            const int output_dim3 = (input_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx >= batch_size * channels * output_dim1 * output_dim2 * output_dim3) return;
            
            const int d3 = idx % output_dim3;
            const int d2 = (idx / output_dim3) % output_dim2;
            const int d1 = (idx / (output_dim3 * output_dim2)) % output_dim1;
            const int ch = (idx / (output_dim3 * output_dim2 * output_dim1)) % channels;
            const int b = idx / (channels * output_dim1 * output_dim2 * output_dim3);
            
            const int input_offset_d1 = -padding + d1 * stride;
            const int input_offset_d2 = -padding + d2 * stride;
            const int input_offset_d3 = -padding + d3 * stride;
            
            scalar_t max_val = -FLT_MAX;
            int max_idx = 0;
            int idx_counter = 0;
            
            for (int k1 = 0; k1 < kernel_size; ++k1) {
                const int in_d1 = input_offset_d1 + dilation * k1;
                if (in_d1 < 0 || in_d1 >= input_dim1) continue;
                
                for (int k2 = 0; k2 < kernel_size; ++k2) {
                    const int in_d2 = input_offset_d2 + dilation * k2;
                    if (in_d2 < 0 || in_d2 >= input_dim2) continue;
                    
                    for (int k3 = 0; k3 < kernel_size; ++k3) {
                        const int in_d3 = input_offset_d3 + dilation * k3;
                        if (in_d3 < 0 || in_d3 >= input_dim3) continue;
                        
                        const int offset = in_d1 * input_dim2 * input_dim3 + in_d2 * input_dim3 + in_d3;
                        const scalar_t val = input[b][ch][in_d1][in_d2][in_d3];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = offset;
                        }
                        idx_counter++;
                    }
                }
            }
            
            output[b][ch][d1][d2][d3] = max_val;
            if (return_indices) indices[b][ch][d1][d2][d3] = max_idx;
        }

        torch::Tensor maxpool3d_forward(torch::Tensor input, 
                                        torch::Tensor output, 
                                        torch::Tensor indices,
                                        int kernel_size, 
                                        int stride, 
                                        int padding, 
                                        int dilation,
                                        bool ceil_mode) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int input_dim1 = input.size(2);
            const int input_dim2 = input.size(3);
            const int input_dim3 = input.size(4);
            const int output_dim1 = (input_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            const int output_dim2 = (input_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            const int output_dim3 = (input_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            
            const int total_elements = batch_size * channels * output_dim1 * output_dim2 * output_dim3;
            
            const int block_size = 256;
            const int grid_size = (total_elements + block_size - 1) / block_size;
            
            AT_DISPATCH_FLOATING_TYPES(input.type(), "maxpool3d_forward", ([&] {
                maxpool3d_forward_kernel<scalar_t><<<grid_size, block_size>>>(
                    input.packed_accessor<scalar_t,5>(),
                    output.packed_accessor<scalar_t,5>(),
                    indices.packed_accessor<int,5>(),
                    batch_size, channels,
                    input_dim1, input_dim2, input_dim3,
                    kernel_size, stride, padding, dilation,
                    ceil_mode
                );
            }));
            
            cudaDeviceSynchronize();
            return output;
        }
        """
        
        maxpool3d_cuda_header = """
        torch::Tensor maxpool3d_forward(torch::Tensor input, 
                                        torch::Tensor output, 
                                        torch::Tensor indices,
                                        int kernel_size, 
                                        int stride, 
                                        int padding, 
                                        int dilation,
                                        bool ceil_mode);
        """
        
        self.maxpool3d_cuda = load_inline(
            name='maxpool3d_cuda',
            cpp_sources=maxpool3d_cuda_header,
            cuda_sources=maxpool3d_cuda_src,
            functions=['maxpool3d_forward'],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get output dimensions
        batch_size, channels, dim1, dim2, dim3 = x.size()
        padding = self.padding
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        output_dim1 = (dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        output_dim2 = (dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        output_dim3 = (dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Initialize output tensors
        output = torch.empty(batch_size, channels, output_dim1, output_dim2, output_dim3, device=x.device)
        indices = torch.empty(batch_size, channels, output_dim1, output_dim2, output_dim3, dtype=torch.int32, device=x.device)

        # Execute custom CUDA kernel
        self.maxpool3d_cuda.maxpool3d_forward(
            x.cuda(),
            output,
            indices,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode
        )

        return output if not self.return_indices else (output, indices)

def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]