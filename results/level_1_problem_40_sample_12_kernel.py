import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super(ModelNew, self).__init__()
        
        # Extract the dimensions
        self.normalized_shape = normalized_shape
        
        # Register parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Define the CUDA kernel for LayerNorm
        layer_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cmath>

        template <typename scalar_t>
        __global__ void layer_norm_kernel(
            const scalar_t* __restrict__ x, scalar_t* __restrict__ y,
            const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
            const int batch_size, const int features, const int dim1, const int dim2,
            const int M, const float eps) {

            extern __shared__ float shared[];

            const int n = batch_size * features * dim1 * dim2;
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;

            // Compute mean and variance
            float sum = 0;
            for (int i = index; i < n; i += blockDim.x * gridDim.x) {
                sum += x[i];
            }
            __shared__ float local_sum;
            local_sum = sum;
            __syncthreads();

            float mean = local_sum / M;
            __syncthreads();

            float var_sum = 0;
            for (int i = index; i < n; i += blockDim.x * gridDim.x) {
                var_sum += (x[i] - mean) * (x[i] - mean);
            }
            __shared__ float local_var_sum;
            local_var_sum = var_sum;
            __syncthreads();

            float var = local_var_sum / M;
            float std = sqrt(var + eps);

            // Normalize and scale/shift
            for (int i = index; i < n; i += blockDim.x * gridDim.x) {
                y[i] = (x[i] - mean) / std * weight[i % M] + bias[i % M];
            }
        }

        torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
            int batch_size, int features, int dim1, int dim2, float eps) {

            const int M = weight.size(0);
            const int N = x.numel() / M;
            const int threads = 256;
            const int blocks = (x.numel() + threads - 1) / threads;

            auto y = torch::empty_like(x);
            auto stream = at::cuda::getCurrentCUDAStream();

            AT_DISPATCH_FLOATING_TYPES(x.type(), "layer_norm_cuda", ([&] {
                layer_norm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    x.data<scalar_t>(), y.data<scalar_t>(),
                    weight.data<scalar_t>(), bias.data<scalar_t>(),
                    batch_size, features, dim1, dim2, M, eps);
            }));

            return y;
        }
        """
        
        layer_norm_cpp = """
        torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
            int batch_size, int features, int dim1, int dim2, float eps);
        """

        # Load the CUDA extension
        self.layer_norm = load_inline(
            name="layer_norm",
            cpp_sources=layer_norm_cpp,
            cuda_sources=layer_norm_source,
            functions=["layer_norm_cuda"],
            verbose=True
        )
        
    def forward(self, x):
        # Ensure inputs are on the correct device
        x = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda()
        
        # Get dimensions
        batch_size, features, dim1, dim2 = x.size()
        eps = 1e-5  # Matching PyTorch's default epsilon
        
        return self.layer_norm.layer_norm_cuda(
            x, weight, bias, batch_size, features, dim1, dim2, eps
        )

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]  # Assuming normalized_shape is (features, dim1, dim2)