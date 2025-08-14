import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn_weight = nn.Parameter(torch.ones(num_features))
        self.bn_bias = nn.Parameter(torch.zeros(num_features))
        self.running_mean = torch.zeros(num_features).cuda()
        self.running_var = torch.ones(num_features).cuda()
        self.eps = 1e-5

        # Define the custom CUDA kernel for fused BatchNorm
        batch_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cub/cub.cuh>

        template <typename T>
        __global__ void batch_norm_forward_kernel(
            const T* input, T* output, 
            const T* weight, const T* bias,
            T* mean, T* invstd,
            const T eps, int N, int C, int spatial_size) {

            extern __shared__ char scratch[];
            T* shared_data = reinterpret_cast<T*>(scratch);
            
            int n = blockIdx.x;
            for (int c = threadIdx.x; c < C; c += blockDim.x) {
                T val = input[n * C * spatial_size + c * spatial_size];
                shared_data[c] = val;
            }
            __syncthreads();

            for (int c = 0; c < C; ++c) {
                T sum = 0;
                for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
                    sum += input[n * C * spatial_size + c * spatial_size + s];
                }
                __shared__ T local_sum[32];
                local_sum[threadIdx.x] = sum;
                __syncthreads();

                sum = 0;
                for (int i = 0; i < blockDim.x; ++i) {
                    sum += local_sum[i];
                }
                __syncthreads();

                mean[n * C + c] = sum / (spatial_size);
                T var = 0;
                for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
                    var += (input[n * C * spatial_size + c * spatial_size + s] - mean[n * C + c]) * (input[n * C * spatial_size + c * spatial_size + s] - mean[n * C + c]);
                }
                __shared__ T local_var[32];
                local_var[threadIdx.x] = var;
                __syncthreads();

                var = 0;
                for (int i = 0; i < blockDim.x; ++i) {
                    var += local_var[i];
                }
                __syncthreads();

                invstd[n * C + c] = 1.0 / sqrt(var / spatial_size + eps);

                output[n * C * spatial_size + c * spatial_size] =
                    (input[n * C * spatial_size + c * spatial_size] - mean[n * C + c]) *
                    (weight[c] * invstd[n * C + c]) + bias[c];
            }
        }

        torch::Tensor batch_norm_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            torch::Tensor running_mean,
            torch::Tensor running_var,
            bool train,
            double momentum,
            torch::Tensor save_mean,
            torch::Tensor save_invstd) {

            int batch_size = input.size(0);
            int C = input.size(1);
            int spatial_size = input.numel() / (batch_size * C);

            dim3 blocks(batch_size);
            dim3 threads(min(256, C)); // Adjust threads based on channels

            batch_norm_forward_kernel<float><<<blocks, threads, threads.y * sizeof(float)>>>(
                input.data_ptr<float>(), output.data_ptr<float>(),
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                save_mean.data_ptr<float>(), save_invstd.data_ptr<float>(),
                eps, batch_size, C, spatial_size);

            // Update running stats
            running_mean = momentum * running_mean + (1 - momentum) * save_mean.mean(0);
            running_var = momentum * running_var + (1 - momentum) * save_invstd.mean(0);

            return output;
        }
        """

        batch_norm_cpp_source = (
            "torch::Tensor batch_norm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, bool train, double momentum, torch::Tensor save_mean, torch::Tensor save_invstd);"
        )

        # Compile the custom CUDA kernel
        self.batch_norm_op = load_inline(
            name="batch_norm_forward",
            cuda_sources=batch_norm_source,
            cpp_sources=batch_norm_cpp_source,
            functions=["batch_norm_forward"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom BatchNorm implementation using CUDA kernel
        if x.is_cuda:
            save_mean = torch.empty_like(x).cuda()
            save_invstd = torch.empty_like(x).cuda()
            return self.batch_norm_op.batch_norm_forward(
                x.cuda(), 
                self.bn_weight.cuda(), 
                self.bn_bias.cuda(), 
                self.running_mean, 
                self.running_var, 
                self.training,
                0.1,
                save_mean,
                save_invstd
            )
        else:
            # Fallback to PyTorch implementation if not on CUDA
            return torch.batch_norm(
                x, 
                self.running_mean, 
                self.running_var, 
                self.bn_weight, 
                self.bn_bias, 
                self.training, 
                0.1, 
                self.eps
            )

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]