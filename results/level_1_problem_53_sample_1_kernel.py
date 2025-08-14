import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Inline CUDA kernel for min reduction along specified dimension
        min_reduction_source = f'''
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <c10/cuda/CUDAMathCompat.h>

        template <typename scalar_t>
        __global__ void min_kernel(scalar_t* out, const scalar_t* in, int64_t dim_size, int64_t outer_size, int64_t inner_size, int64_t reduce_dim) {{
            int block_idx = blockIdx.x;
            int thread_idx = threadIdx.x;
            int block_offset = block_idx * blockDim.x;
            int index = block_offset + thread_idx;

            if (index < outer_size * inner_size) {{
                int in_offset = index * dim_size;
                scalar_t min_val = in[in_offset + thread_idx];
                for (int i = 1; i < dim_size; ++i) {{
                    min_val = min(min_val, in[in_offset + i]);
                }}
                out[index] = min_val;
            }}
        }}

        torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t reduce_dim) {{
            auto input_shape = input.sizes();
            int64_t batch_size = input_shape[0];
            int64_t dim_size = input_shape[reduce_dim];
            int64_t outer_size = 1;
            int64_t inner_size = 1;

            // Compute outer and inner dimensions for non-reduced axes
            for (int64_t i = 0; i < reduce_dim; ++i) {{
                outer_size *= input_shape[i];
            }}
            for (int64_t i = reduce_dim + 1; i < input.dim(); ++i) {{
                inner_size *= input_shape[i];
            }}

            int64_t total_elements = outer_size * inner_size;
            int threads_per_block = 256;
            int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

            auto output = torch::empty({{outer_size * inner_size}}, input.options());

            min_kernel<<<num_blocks, threads_per_block>>>(
                output.data_ptr<float>(),
                input.data_ptr<float>(),
                dim_size,
                outer_size,
                inner_size,
                reduce_dim
            );

            cudaDeviceSynchronize();
            return output.view({{batch_size, inner_size}});
        }}
        '''

        min_reduction_cpp_source = '''
        torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t reduce_dim);
        '''

        self.min_reduction = load_inline(
            name='min_reduction',
            cpp_sources=min_reduction_cpp_source,
            cuda_sources=min_reduction_source,
            functions=['min_reduction_cuda'],
            verbose=True,
            with_cuda=True
        )

    def forward(self, x: torch.Tensor):
        return self.min_reduction.min_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda() if torch.cuda.is_available() else torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [torch.randint(1, (1,))]  # Adjust according to model requirements