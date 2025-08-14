import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Inline CUDA kernel for RMSNorm
        rmsnorm_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void rmsnorm_forward_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y,
                                               const float eps, int batch_size, int features, int dim, int total_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_size) return;

            // Compute the feature dimension index
            int feature_idx = (idx / dim) % features;
            int feature_base = (idx / (features * dim)) * features * dim;

            // Compute mean of squares along feature dim
            extern __shared__ volatile scalar_t shared_data[];
            scalar_t val = x[idx];
            scalar_t square_sum = val * val;
            for (int i = feature_base; i < feature_base + features * dim; i += features) {
                square_sum += x[i + (idx % dim)] * x[i + (idx % dim)];
            }

            // Use block reduction for sum
            __shared__ volatile float block_sums[32];
            int tid = threadIdx.x;
            block_sums[tid] = square_sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    block_sums[tid] += block_sums[tid + s];
                }
                __syncthreads();
            }

            float mean = 0;
            if (tid == 0) {
                mean = block_sums[0] / (features * dim);
            }
            __syncthreads();

            scalar_t rms = 1.0 / sqrt(mean + eps);
            y[idx] = x[idx] * rms;
        }

        torch::Tensor rmsnorm_forward(torch::Tensor x, float eps) {
            const auto batch_size = x.size(0);
            const auto features = x.size(1);
            const auto dim = x.size(2) * x.size(3); // Handle 4D tensors
            const auto total_size = x.numel();

            const int block_size = 256;
            const int num_blocks = (total_size + block_size - 1) / block_size;

            auto y = torch::empty_like(x);

            rmsnorm_forward_kernel<float><<<num_blocks, block_size, block_size * sizeof(float)>>>(
                x.data_ptr<float>(), y.data_ptr<float>(), eps, batch_size, features, dim, total_size
            );

            return y;
        }
        """

        rmsnorm_cpp_source = (
            "torch::Tensor rmsnorm_forward(torch::Tensor x, float eps);"
        )

        # Compile the kernel
        rmsnorm_module = load_inline(
            name="rmsnorm",
            cpp_sources=rmsnorm_cpp_source,
            cuda_sources=rmsnorm_source,
            functions=["rmsnorm_forward"],
            verbose=True
        )

        self.rmsnorm_forward = rmsnorm_module.rmsnorm_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm_forward(x, self.eps).cuda()