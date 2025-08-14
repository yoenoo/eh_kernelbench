import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Load the custom CUDA kernel for mean reduction
        self.mean_reduction = load_inline(
            name="mean_reduction_cuda",
            cpp_sources=f"""
                torch::Tensor mean_reduction_cuda(torch::Tensor x, int64_t dim);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>

                template <typename scalar_t>
                __global__ void mean_kernel(scalar_t* x_data, scalar_t* out_data, int64_t outer_size, int64_t inner_size, int64_t dim) {
                    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= outer_size) return;

                    scalar_t sum = 0.0;
                    for (int64_t i = 0; i < inner_size; ++i) {
                        int64_t offset = idx * inner_size + i;
                        sum += x_data[offset];
                    }
                    out_data[idx] = sum / inner_size;
                }

                torch::Tensor mean_reduction_cuda(torch::Tensor x, int64_t dim) {{
                    auto x_dims = x.sizes().vec();
                    int64_t dim_size = x.size(dim);
                    auto output_shape = x.sizes().vec();
                    output_shape.erase(output_shape.begin() + dim);

                    auto out = torch::empty(output_shape, x.options());
                    int64_t threads = 256;
                    int64_t blocks = out.numel();

                    auto stream = at::cuda::getCurrentCUDAStream();
                    if (dim == 0) {{
                        // Special handling for dimension 0 (batch)
                        throw std::runtime_error("Non contiguous dimensions or dim0 not currently implemented");
                    }}
                    // Calculate contiguous dimensions
                    int64_t outer_size = 1;
                    for (int64_t i = 0; i < dim; ++i) {{
                        outer_size *= x_dims[i];
                    }}
                    int64_t inner_size = x_dims[dim];
                    auto x_contig = x.contiguous();
                    mean_kernel<float><<<blocks, threads, 0, stream>>>(
                        static_cast<float*>(x_contig.data_ptr()),
                        static_cast<float*>(out.data_ptr()),
                        outer_size,
                        inner_size,
                        dim
                    );
                    return out;
                }}
            """,
            functions=["mean_reduction_cuda"],
            verbose=True,
            with_cuda=True
        )

    def forward(self, x: torch.Tensor):
        # Use the custom kernel for mean calculation
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)

def get_inputs():
    # Generate inputs as before
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]

batch_size = 128
dim1 = 4096
dim2 = 4095