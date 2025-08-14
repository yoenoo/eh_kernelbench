import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        # Define custom CUDA kernel for layer normalization
        layer_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <c10/macros/Macros.h>
        #include <ATen/cuda/CUDAContext.h>

        template <typename scalar_t>
        __global__ void layer_norm_forward_kernel(const scalar_t* __restrict__ x,
                                                  scalar_t* __restrict__ y,
                                                  const scalar_t* __restrict__ weight,
                                                  const scalar_t* __restrict__ bias,
                                                  const int N,
                                                  const int D,
                                                  const float eps) {
            const int offset = blockIdx.x * D + threadIdx.x;
            const int stride = gridDim.x * D;

            scalar_t mean = 0.0;
            scalar_t var = 0.0;

            // Compute mean
            for (int i = offset; i < N * D; i += stride) {
                mean += x[i];
            }
            __shared__ scalar_t shared_mean[1024];
            shared_mean[threadIdx.x] = mean;
            __syncthreads();

            if (threadIdx.x == 0) {
                scalar_t sum = 0.0;
                for (int i = 0; i < blockDim.x; i++) {
                    sum += shared_mean[i];
                }
                shared_mean[0] = sum / D;
            }
            __syncthreads();

            scalar_t mu = shared_mean[0];

            // Compute variance
            for (int i = offset; i < N * D; i += stride) {
                var += (x[i] - mu) * (x[i] - mu);
            }
            __shared__ scalar_t shared_var[1024];
            shared_var[threadIdx.x] = var;
            __syncthreads();

            if (threadIdx.x == 0) {
                scalar_t sum = 0.0;
                for (int i = 0; i < blockDim.x; i++) {
                    sum += shared_var[i];
                }
                shared_var[0] = sqrt(sum / D + eps);
            }
            __syncthreads();

            scalar_t sigma = shared_var[0];

            // Compute output
            for (int i = offset; i < N * D; i += stride) {
                y[i] = (x[i] - mu) / sigma * weight[threadIdx.x] + bias[threadIdx.x];
            }
        }

        torch::Tensor layer_norm_forward_cuda(torch::Tensor x,
                                             torch::Tensor weight,
                                             torch::Tensor bias,
                                             float eps) {
            const int N = x.size(0);
            const int D = x.size(1);
            auto y = torch::empty_like(x);

            const int block_size = 256;
            const int num_blocks = (N * D + block_size - 1) / block_size;

            AT_DISPATCH_FLOATING_TYPES(x.type(), "layer_norm_forward", ([&] {
                layer_norm_forward_kernel<scalar_t><<<num_blocks, block_size>>>(
                    x.data<scalar_t>(),
                    y.data<scalar_t>(),
                    weight.data<scalar_t>(),
                    bias.data<scalar_t>(),
                    N,
                    D,
                    eps);
            }));

            cudaDeviceSynchronize();
            return y;
        }
        """

        layer_norm_cpp_source = """
        torch::Tensor layer_norm_forward_cuda(torch::Tensor x,
                                             torch::Tensor weight,
                                             torch::Tensor bias,
                                             float eps);
        """

        # Compile the custom kernel
        self.layer_norm = load_inline(
            name="layer_norm",
            cpp_sources=layer_norm_cpp_source,
            cuda_sources=layer_norm_source,
            functions=["layer_norm_forward_cuda"],
            verbose=True,
            extra_cflags=["-DWITH_CUDA"],
            extra_ldflags=[""]
        )
        self.eps = 1e-5  # Default epsilon value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm.layer_norm_forward_cuda(x, self.weight, self.bias, self.eps)