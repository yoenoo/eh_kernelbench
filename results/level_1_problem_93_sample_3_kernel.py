import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        # Define and compile the custom CUDA kernel
        cumsum_masked_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void masked_cumsum_kernel(const scalar_t* x, const bool* mask, scalar_t* out, int64_t* dims, int64_t* strides, int64_t total_elements, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements) return;

            int pos = idx;
            int offset = 0;
            scalar_t sum = 0;
            int dim_size = dims[dim];
            for (int i = 0; i < dim_size; i++) {
                // Calculate the position along the current dimension
                int current_pos = pos;
                if (mask[current_pos]) {
                    sum += x[current_pos];
                }
                out[current_pos] = sum;
                // Move to next position along the dimension
                current_pos += strides[dim];
                if (current_pos >= total_elements) {
                    break;
                }
                pos = current_pos;
            }
        }

        torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim) {
            auto x_size = x.sizes();
            auto x_stride = x.stride();
            auto dims = x_size.vec();
            auto strides = x_stride.vec();

            auto out = torch::empty_like(x);
            int total_elements = x.numel();

            const int block_size = 256;
            const int num_blocks = (total_elements + block_size - 1) / block_size;

            // Launch kernel
            AT_DISPATCH_ALL_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
                masked_cumsum_kernel<scalar_t><<<num_blocks, block_size>>>(
                    x.data<scalar_t>(),
                    mask.data<bool>(),
                    out.data<scalar_t>(),
                    dims.data(),
                    strides.data(),
                    total_elements,
                    dim);
            }));

            cudaDeviceSynchronize();
            return out;
        }
        """

        cumsum_masked_cpp_source = "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim);"

        self.masked_cumsum = load_inline(
            name="masked_cumsum",
            cpp_sources=cumsum_masked_cpp_source,
            cuda_sources=cumsum_masked_source,
            functions=["masked_cumsum_cuda"],
            verbose=False,
            with_cuda=True,
        )

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x.cuda(), mask.cuda(), self.dim)