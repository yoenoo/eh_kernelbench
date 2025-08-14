import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

        # Define and compile the custom CUDA kernel for LeakyReLU
        leaky_relu_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void leaky_relu_kernel(const float* input, float* output, const float negative_slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float val = input[idx];
                output[idx] = val > 0.0f ? val : val * negative_slope;
            }
        }

        torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
            auto output = torch::empty_like(input);
            int size = input.numel();
            
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            leaky_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), 
                                                         output.data_ptr<float>(), 
                                                         negative_slope, 
                                                         size);
            return output;
        }
        """

        leaky_relu_cpp_source = "torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope);"

        self.custom_leaky_relu = load_inline(
            name="custom_leaky_relu",
            cpp_sources=leaky_relu_cpp_source,
            cuda_sources=leaky_relu_source,
            functions=["leaky_relu_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_leaky_relu.leaky_relu_cuda(x, self.negative_slope)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []