cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ReverseCumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        size = input.size(ctx.dim)
        output = torch.empty_like(input)
        
        # Define CUDA kernel for reverse cumulative sum
        kernel_src = f"""
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        __global__ void reverse_cumsum_fwd(const float* input, float* output, int dim_size, int batch_size, int dim) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int size = dim_size;
            int outer = idx / size;
            int inner = idx % size;

            if (idx >= batch_size * dim_size) return;
            // Reverse index along dimension
            int rev_idx = size - 1 - inner;
            float sum = 0.0;
            for (int i = 0; i <= rev_idx; ++i) {{
                int pos = outer * size + (size - 1 - i);
                sum += input[pos];
            }}
            output[idx] = sum;
        }}

        at::Tensor reverse_cumsum_forward(at::Tensor input, int dim) {{
            const int dims[] = {{input.size(0), input.size(1)}};
            int elements = input.numel();
            const int threads_per_block = 256;
            int blocks_per_grid = (elements + threads_per_block - 1) / threads_per_block;
            auto output = at::empty_like(input);
            reverse_cumsum_fwd<<<blocks_per_grid, threads_per_block>>>(
                input.data_ptr<float>(), output.data_ptr<float>(),
                dims[dim], dims[1 - dim], dim);
            return output;
        }}
        """

        # Load and compile the CUDA kernel
        cudarmodule = load_inline(
            name="reverse_cumsum",
            cpp_sources="",
            cuda_sources=kernel_src,
            functions=["reverse_cumsum_forward"],
            verbose=True
        )
        ctx.save_for_backward(input)
        return cudarmodule.reverse_cumsum_forward(input, dim)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        dim = ctx.dim
        grad_input = torch.zeros_like(input)
        # Implement backward kernel similarly or use PyTorch's functions
        # For simplicity using PyTorch here but can be optimized with a kernel
        grad_input = grad_output.flip(dim).cumsum(dim).flip(dim)
        return grad_input, None

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return ReverseCumsumFunction.apply(x, self.dim)