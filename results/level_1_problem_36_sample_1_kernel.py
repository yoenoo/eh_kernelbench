import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Define and load the custom CUDA kernel for RMSNorm
        rmsnorm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        template <typename scalar_t>
        __global__ void rmsnorm_forward_kernel(
            const scalar_t* __restrict__ x,
            scalar_t* __restrict__ y,
            int batch_size,
            int num_features,
            int dim,
            float eps
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * dim) {
                return;
            }

            // Calculate the position along the feature dimension
            int batch_dim = batch_size * dim;
            int feature_offset = (idx % batch_dim) / dim;

            // Compute squared elements for the mean
            scalar_t sum_sq = 0;
            for (int f = 0; f < num_features; ++f) {
                int pos = idx + f * batch_dim;
                scalar_t val = x[pos];
                sum_sq += val * val;
            }

            // Use block reduction for sum
            __shared__ scalar_t shared_sum_sq[32];
            int tid = threadIdx.x;
            shared_sum_sq[tid] = sum_sq;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_sum_sq[tid] += shared_sum_sq[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                shared_sum_sq[0] = sqrt(shared_sum_sq[0] / num_features + eps);
            }
            __syncthreads();

            // Normalize each element
            for (int f = 0; f < num_features; ++f) {
                int pos = idx + f * batch_dim;
                y[pos] = x[pos] / shared_sum_sq[0];
            }
        }

        torch::Tensor rmsnorm_forward_cuda(torch::Tensor x, float eps) {
            const int batch_size = x.size(0);
            const int num_features = x.size(1);
            const int dim = x.size(2) * x.size(3);

            auto y = torch::empty_like(x);

            const int threads = 256;
            const int elements = batch_size * dim;
            const int blocks = (elements + threads - 1) / threads;

            // Launch kernel
            AT_DISPATCH_FLOATING_TYPES(x.type(), "rmsnorm_forward", ([&] {
                rmsnorm_forward_kernel<scalar_t><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    batch_size,
                    num_features,
                    dim,
                    eps
                );
            }));

            cudaDeviceSynchronize();
            return y;
        }
        """

        rmsnorm_forward_cpp = "torch::Tensor rmsnorm_forward_cuda(torch::Tensor x, float eps);"

        # Compile the kernel
        self.rmsnorm = load_inline(
            name="rmsnorm",
            cpp_sources=rmsnorm_forward_cpp,
            cuda_sources=rmsnorm_source,
            functions=["rmsnorm_forward_cuda"],
            verbose=True,
            with_cuda=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if CUDA is available and move tensors to GPU
        if x.is_cuda:
            return self.rmsnorm.rmsnorm_forward_cuda(x, self.eps)
        else:
            # Fallback to PyTorch for CPU (though model is designed for CUDA)
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
            return x / rms

# Ensure inputs are moved to GPU in get_inputs
def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]