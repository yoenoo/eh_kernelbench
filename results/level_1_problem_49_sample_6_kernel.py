import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Define custom CUDA kernel for max reduction
        max_reduction_source = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <limits>

        template <typename scalar_t>
        __global__ void max_kernel(const scalar_t* input, scalar_t* output,
                                  int dim_size, int batch_size, int other_dim, int64_t dim) {{
            const int batch_idx = blockIdx.x;
            const int other_idx = blockIdx.y;
            const int idx = batch_idx * other_dim * dim_size + other_idx * dim_size + threadIdx.x;

            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            if (dim == 0) {{
                for (int d = 0; d < dim_size; ++d) {{
                    int pos = d * other_dim * batch_size + other_idx * batch_size + batch_idx;
                    scalar_t val = input[pos];
                    if (val > max_val) max_val = val;
                }}
            }} else if (dim == 1) {{
                for (int d = 0; d < dim_size; ++d) {{
                    int pos = batch_idx * dim_size * other_dim + d * other_dim + other_idx;
                    scalar_t val = input[pos];
                    if (val > max_val) max_val = val;
                }}
            }} else {{
                // Assume dim is 2 (last dimension) for simplicity
                for (int d = 0; d < dim_size; ++d) {{
                    int pos = batch_idx * other_dim * dim_size + other_idx * dim_size + d;
                    scalar_t val = input[pos];
                    if (val > max_val) max_val = val;
                }}
            }}
            if (threadIdx.x == 0) {{
                output[batch_idx * other_dim + other_idx] = max_val;
            }}
        }}

        torch::Tensor max_reduction_cuda(torch::Tensor input, int64_t dim) {{
            const int64_t batch_size = input.size(0);
            const int64_t dim_size = input.size(dim);
            const int64_t other_dim = (dim == 0) ? input.size(2) : (dim == 1) ? input.size(2) : input.size(1);

            auto output = torch::empty({{batch_size, other_dim}}, input.options());

            dim3 block(dim_size);
            dim3 grid(batch_size, other_dim);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "max_reduction_cuda", ([&] {{
                max_kernel<scalar_t><<<grid, block>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    dim_size, batch_size, other_dim, dim);
            }}));

            cudaDeviceSynchronize();
            return output;
        }}
        """
        max_reduction_cpp_source = """
        torch::Tensor max_reduction_cuda(torch::Tensor input, int64_t dim);
        """

        # Dynamically compile the kernel
        self.max_reduction = load_inline(
            name="max_reduction",
            cpp_sources=max_reduction_cpp_source,
            cuda_sources=max_reduction_source,
            functions=["max_reduction_cuda"],
            verbose=False,
            with_cuda=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)