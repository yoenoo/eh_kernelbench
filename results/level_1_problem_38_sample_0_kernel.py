import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
        # Define the custom CUDA kernel for L1 normalization
        l1_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void l1_norm_kernel(const float* x, float* out, int batch_size, int dim) {
            int batch = blockIdx.x;
            int idx = threadIdx.x + blockDim.x * batch;
            
            // Each thread handles one element of x
            if (idx < batch_size * dim) {
                int d = idx % dim; // current dimension index
                int b = idx / dim; // current batch index
                
                // Compute the absolute value and sum
                float sum = 0.0;
                for (int i = 0; i < dim; ++i) {
                    sum += abs(x[b * dim + i]);
                }
                
                // Normalize and write to output
                if (sum != 0) {
                    out[idx] = x[idx] / sum;
                } else {
                    // Avoid division by zero, set to zero
                    out[idx] = 0.0;
                }
            }
        }
        
        torch::Tensor l1_norm_cuda(torch::Tensor x) {
            const auto batch_size = x.size(0);
            const auto dim = x.size(1);
            
            auto out = torch::empty_like(x);
            
            const int block_size = 256;
            const int num_blocks = (batch_size * dim + block_size - 1) / block_size;
            dim3 grid(num_blocks);
            dim3 block(block_size);
            
            l1_norm_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);
            
            return out;
        }
        """
        
        # Compile the CUDA kernel
        self.l1_norm = load_inline(
            name="l1_norm",
            cuda_sources=l1_norm_source,
            functions=["l1_norm_cuda"],
            verbose=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)