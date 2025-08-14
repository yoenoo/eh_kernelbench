import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        # Define and load the custom CUDA kernel for argmax
        argmax_cuda_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cuda_fp16.h>
        
        template <typename scalar_t>
        __global__ void custom_argmax_kernel(const scalar_t* input, int64_t* output,
                                            int dim, int outer_dim, int inner_dim) {
            int batch_idx = blockIdx.x;
            int inner_idx = threadIdx.x;
            
            // Compute the index along the specified dimension
            for (int i = 0; i < outer_dim; ++i) {
                int index = batch_idx * outer_dim * inner_dim + i * inner_dim + inner_idx;
                if (i == 0) {
                    output[batch_idx * inner_dim + inner_idx] = 0;
                    scalar_t max_val = input[index];
                    // Find maximum value index along the dimension
                    for (int j = 1; j < inner_dim; ++j) {
                        int current_idx = batch_idx * outer_dim * inner_dim + j * inner_dim + inner_idx;
                        if (input[current_idx] > max_val) {
                            max_val = input[current_idx];
                            output[batch_idx * inner_dim + inner_idx] = j;
                        }
                    }
                }
            }
        }

        torch::Tensor custom_argmax_cuda(torch::Tensor input, int dim) {
            auto input_shape = input.sizes().vec();
            int outer_dim = input.size(dim);
            int inner_dim = 1;
            for (int i = 0; i < input.dim(); ++i) {
                if (i != dim) {
                    inner_dim *= input.size(i);
                }
            }
            
            auto output = torch::empty({inner_dim}, input.options().dtype(torch::kLong)).cuda();
            int threads = input.size(input.dim() - 1);
            int blocks = 1; // Assuming batch is first dimension
            for (int i = 0; i < input.dim(); ++i) {
                if (i != 0 && i != dim) {
                    blocks *= input.size(i);
                }
            }
            
            // Launch kernel with appropriate dimensions
            AT_DISPATCH_ALL_TYPES(input.scalar_type(), "custom_argmax_cuda", ([&] {
                custom_argmax_kernel<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(), output.data<int64_t>(),
                    dim, outer_dim, inner_dim);
            }));
            
            return output.view(input_shape).narrow(dim, 0, 1).squeeze(dim);
        }
        """
        
        # Load the CUDA kernel
        self.argmax_op = load_inline(
            name='custom_argmax',
            cuda_sources=argmax_cuda_source,
            functions=['custom_argmax_cuda'],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_op.custom_argmax_cuda(x, self.dim)