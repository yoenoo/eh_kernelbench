import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Define and load the custom CUDA kernel
        self.cuda_cumsum = load_inline(
            name='exclusive_cumsum',
            cuda_sources=f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void exclusive_cumsum_kernel(scalar_t *out, const scalar_t *x, int64_t size, int64_t dim_size, int64_t before, int64_t after) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    int pos = idx;
    int d = pos / dim_size;
    pos %= dim_size;
    if (pos == 0) {{
        out[idx] = 0;
    }} else {{
        out[idx] = out[idx - 1];
        if (pos > 0) {{
            out[idx] += x[idx - before - after];
        }}
    }}
}}

int64_t calc_block_count(int64_t size) {{
    int64_t block_size = 256;
    return (size + block_size - 1) / block_size;
}}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int dim) {{
    auto dims = x.sizes().vec();
    auto size = x.numel();
    auto dim_size = dims[dim];
    auto before = 1;
    for (int i = 0; i < dim; i++) {{
        before *= dims[i];
    }}
    auto after = 1;
    for (int i = dim + 1; i < dims.size(); i++) {{
        after *= dims[i];
    }}

    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = calc_block_count(size);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "exclusive_cumsum_cuda", ([&] {{
        exclusive_cumsum_kernel<scalar_t><<<num_blocks, block_size>>>(
            out.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            size,
            dim_size,
            before,
            after
        );
    }}));

    return out;
}}
""",
            cuda_headers="""
#include <ATen/cuda/CUDAContext.h>
""",
            functions=["exclusive_cumsum_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.cuda_cumsum.exclusive_cumsum_cuda(x, self.dim)