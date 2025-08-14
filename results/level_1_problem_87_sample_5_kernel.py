import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void pointwise_conv_kernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     int batch_size,
                                     int in_channels,
                                     int out_channels,
                                     int height,
                                     int width,
                                     const scalar_t* __restrict__ bias) {
    int total_elements = batch_size * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / (height * width);
    int spatial_idx = idx % (height * width);
    
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        scalar_t sum = 0;
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            sum += input[batch_idx * in_channels * height * width + in_ch * height * width + spatial_idx] * 
                   weight[out_ch * in_channels + in_ch];
        }
        if (bias) {
            sum += bias[out_ch];
        }
        output[batch_idx * out_channels * height * width + out_ch * height * width + spatial_idx] = sum;
    }
}

torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height = input.size(2);
    const int width = input.size(3);
    
    auto output = torch::empty({batch_size, out_channels, height, width}, input.options());
    
    int total_elements = batch_size * height * width;
    const int block_size = BLOCK_SIZE;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (bias.defined()) {
        pointwise_conv_kernel<float><<<num_blocks, block_size, 0, stream>>>(input.data_ptr<float>(),
                                                                           weight.data_ptr<float>(),
                                                                           output.data_ptr<float>(),
                                                                           batch_size,
                                                                           in_channels,
                                                                           out_channels,
                                                                           height,
                                                                           width,
                                                                           bias.data_ptr<float>());
    } else {
        pointwise_conv_kernel<float><<<num_blocks, block_size, 0, stream>>>(input.data_ptr<float>(),
                                                                           weight.data_ptr<float>(),
                                                                           output.data_ptr<float>(),
                                                                           batch_size,
                                                                           in_channels,
                                                                           out_channels,
                                                                           height,
                                                                           width,
                                                                           nullptr);
    }
    
    return output;
}
"""

pointwise_conv_cpp = """
torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=[pointwise_conv_cpp],
    cuda_sources=[pointwise_conv_source],
    functions=["pointwise_conv_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias is not None and self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        self.pointwise_conv_op = pointwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv_op.pointwise_conv_cuda(x, self.weight, self.bias if self.bias is not None else torch.Tensor())