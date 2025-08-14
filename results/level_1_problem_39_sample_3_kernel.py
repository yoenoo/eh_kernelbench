import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the custom CUDA kernel for L2 normalization
        l2norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void l2_norm_kernel(const float* x, float* out, const int batch_size, const int dim) {
            int batch_idx = blockIdx.x;
            int elem_idx = threadIdx.x;

            __shared__ float sum_sqr[256]; // Use shared memory for reducing sum of squares

            if (elem_idx < dim) {
                float val = x[batch_idx * dim + elem_idx];
                sum_sqr[threadIdx.x] = val * val;
            } else {
                sum_sqr[threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Reduce sum of squares using block reduction
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + stride];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                float norm = sqrt(sum_sqr[0]);
                if (norm > 1e-12) {
                    norm = 1.0f / norm;
                }
                for (int i = 0; i < dim; i += blockDim.x) {
                    int idx = i + threadIdx.x;
                    if (idx < dim) {
                        out[batch_idx * dim + idx] = x[batch_idx * dim + idx] * norm;
                    }
                }
            }
        }

        torch::Tensor l2_norm_cuda(torch::Tensor x) {
            const int batch_size = x.size(0);
            const int dim = x.size(1);

            auto out = torch::empty_like(x);
            
            // Block size per batch is up to dim, but limited by maximum threads per block
            int block_size = std::min(dim, 1024);
            dim3 blocks(batch_size);
            dim3 threads(block_size);

            l2_norm_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);
            
            return out;
        }
        """
        
        l2norm_h_source = """
        torch::Tensor l2_norm_cuda(torch::Tensor x);
        """
        
        self.l2norm_kernel = load_inline(
            name="l2_norm",
            cpp_sources=l2norm_h_source,
            cuda_sources=l2norm_source,
            functions=["l2_norm_cuda"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2norm_kernel.l2_norm_cuda(x)