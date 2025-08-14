import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Define the custom CUDA kernel for exclusive cumsum
        exclusive_cumsum_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void exclusive_cumsum_kernel(const scalar_t* input, scalar_t* output, const int64_t* dims, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_size = 1;
            for (int i = 0; i < dim; i++) {
                total_size *= dims[i];
            }
            int stride = 1;
            for (int i = dim + 1; i < dims.size(); i++) {
                stride *= dims[i];
            }
            for (int pos = idx; idx < total_size; idx += blockDim.x * gridDim.x) {
                int base = idx * stride * dims[dim];
                for (int i = 0; i < stride; i++) {
                    scalar_t sum = 0;
                    for (int j = 0; j < dims[dim] - 1; j++) {
                        int offset = base + i + j * stride;
                        output[offset] = sum;
                        sum += input[offset];
                    }
                    output[base + dims[dim] * stride - 1 + i] = sum;
                }
            }
        }

        torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim) {
            const auto dims = input.sizes().vec();
            auto output = torch::empty_like(input);
            int n_threads = 256;
            int n_blocks = (input.numel() + n_threads - 1) / n_threads;
            auto stream = at::cuda::getCurrentCUDAStream();
            AT_DISPATCH_ALL_TYPES(input.type(), "exclusive_cumsum_cuda", ([&] {
                exclusive_cumsum_kernel<scalar_t><<<n_blocks, n_threads, 0, stream>>>(
                    input.data<scalar_t>(), output.data<scalar_t>(), dims.data(), dim);
            }));
            return output;
        }
        """

        exclusive_cumsum_header = """
        torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim);
        """

        # Compile the CUDA kernel
        exclusive_cumsum = load_inline(
            name="exclusive_cumsum",
            cpp_sources=exclusive_cumsum_header,
            cuda_sources=exclusive_cumsum_source,
            functions=["exclusive_cumsum_cuda"],
            verbose=True
        )
        self.exclusive_cumsum = exclusive_cumsum

    def forward(self, x):
        return self.exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)