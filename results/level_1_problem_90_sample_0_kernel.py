import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    """
    Optimized version of the cumulative product model using a custom CUDA kernel.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Load the custom CUDA kernel
        self.cumprod_cuda = load_inline(
            name="cumprod_cuda",
            cpp_sources="""
                torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <cmath>

                template <typename scalar_t>
                __global__ void cumprod_kernel(
                    const scalar_t* __restrict__ data,
                    scalar_t* __restrict__ out,
                    int64_t size,
                    int64_t stride,
                    int64_t dim_size
                ) {{
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= size) return;

                    int dim_idx = idx / stride;
                    int remainder = idx % stride;
                    out[idx] = 1;
                    for (int i = 0; i <= dim_idx; ++i) {{
                        out[idx] *= data[remainder + (i * stride)];
                    }}
                }}

                torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim) {{
                    auto dims = x.sizes().vec();
                    int64_t total_size = x.numel();
                    int64_t dim_size = dims[dim];
                    int64_t stride = 1;
                    for (int64_t i = dim + 1; i < dims.size(); ++i) {{
                        stride *= dims[i];
                    }}

                    torch::Tensor out = torch::empty_like(x);
                    const int block_size = 256;
                    const int grid_size = (total_size + block_size - 1) / block_size;
                    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "cumprod_cuda", ([&] {{
                        cumprod_kernel<scalar_t>
                            <<<grid_size, block_size>>>(
                                x.data_ptr<scalar_t>(),
                                out.data_ptr<scalar_t>(),
                                total_size,
                                stride,
                                dim_size
                            );
                    }}));
                    return out;
                }}
            """,
            functions=["cumprod_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.cumprod_cuda.cumprod_cuda(x, self.dim)