import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

        # Define the custom CUDA kernel for reverse cumulative sum
        reverse_cumsum_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, const int dim, const int size,
                                              const int inner_dim, const int outer_dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            int outer = idx / inner_dim;
            int pos = (idx % inner_dim);
            int original_pos = inner_dim - 1 - pos;

            scalar_t sum = 0;
            for (int i = 0; i <= pos; i++) {
                int input_idx = outer * inner_dim + (inner_dim - 1 - i);
                sum += input[input_idx];
            }
            output[idx] = sum;
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim) {
            const int64_t size = input.numel();
            auto output = torch::empty_like(input);
            int batch_size = input.size(0);
            int inner_dim = input.size(dim);

            dim3 threads(1024);
            dim3 blocks((size + threads.x - 1) / threads.x);

            int inner = 1;
            for (int i = 0; i < dim; i++) inner *= input.size(i);
            int outer = size / inner;

            reverse_cumsum_kernel<float><<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                dim,
                size,
                inner_dim,
                outer
            );

            return output;
        }
        """

        # Compile the inline CUDA code
        reverse_cumsum = load_inline(
            name="reverse_cumsum",
            cpp_sources="",
            cuda_sources=reverse_cumsum_source,
            functions=["reverse_cumsum_cuda"],
            verbose=True
        )

        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)