import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Define the custom CUDA kernel for argmin operation
        argmin_source = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <limits>

        __global__ void argmin_kernel(const float* data, int* indices, int batch_size, int dim1, int dim2, int dim) {{
            int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (batch_idx >= batch_size) return;

            int offset = batch_idx * dim1 * dim2;
            for (int i = 0; i < dim == 0 ? dim1 : dim == 1 ? dim2 : 1; ++i) {{
                int idx = 0;
                float min_val = std::numeric_limits<float>::max();
                for (int j = 0; j < (dim == 0 ? dim1 : dim == 1 ? dim2 : 1); ++j) {{
                    int pos = offset + (dim == 0 ? j * dim2 + i : i * dim2 + j);
                    float val = data[pos];
                    if (val < min_val) {{
                        min_val = val;
                        idx = j;
                    }}
                }}
                indices[batch_idx * (dim == 0 ? dim1 : dim2) + i] = idx;
            }}
        }}

        torch::Tensor argmin_cuda(torch::Tensor data, int dim_val) {{
            int batch_size = data.size(0);
            int dim1_val = data.size(1);
            int dim2_val = data.size(2);
            int output_size = dim_val == 0 ? batch_size * dim2_val : batch_size * dim1_val;

            auto indices = torch::empty(output_size, data.options().dtype(torch::kInt32));

            int threads_per_block = 256;
            int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

            argmin_kernel<<<blocks_per_grid, threads_per_block>>>(
                data.data_ptr<float>(),
                indices.data_ptr<int>(),
                batch_size,
                dim1_val,
                dim2_val,
                dim_val
            );

            return indices.view({{batch_size, dim_val == 0 ? dim2_val : dim1_val}});
        }}
        """
        argmin_cpp_src = "torch::Tensor argmin_cuda(torch::Tensor data, int dim_val);"

        # Compile the custom kernel
        self.argmin_cuda = load_inline(
            name="custom_argmin",
            cpp_sources=argmin_cpp_src,
            cuda_sources=argmin_source,
            functions=["argmin_cuda"],
            verbose=False,
        )

    def forward(self, x):
        return self.argmin_cuda.argmin_cuda(x, self.dim).to(x.device)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x.cuda()]

def get_init_inputs():
    return [dim]