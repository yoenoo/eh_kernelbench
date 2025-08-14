import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        # Maintain the original GroupNorm parameters for compatibility and initialization
        self.num_groups = num_groups
        self.num_channels = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5  # Default epsilon value, adjust if needed

        # Load the custom CUDA kernel for group normalization
        group_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cmath>

        template <typename scalar_t>
        __global__ void group_norm_kernel(
            scalar_t* x, scalar_t* weight, scalar_t* bias, scalar_t* output,
            const int batch_size, const int num_groups, const int channels_per_group,
            const int spatial_size, const float eps) {

            const int group_id = blockIdx.z;
            const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
            const int spatial_idx = blockIdx.w * blockDim.z + threadIdx.z;

            if (batch_idx >= batch_size || channel_idx >= channels_per_group || spatial_idx >= spatial_idx) {
                return;
            }

            const int group_channels_start = group_id * channels_per_group;
            const int channel = group_channels_start + channel_idx;
            const int x_index = batch_idx * (num_groups * channels_per_group * spatial_size) +
                                group_channels_start * spatial_size + channel_idx * spatial_size + spatial_idx;
            const scalar_t x_val = x[x_index];

            // Compute mean and variance for the group
            extern __shared__ scalar_t shared[];
            scalar_t* buffer = shared;
            scalar_t sum = 0.0;

            for (int b = 0; b < batch_size; ++b) {
                for (int c = 0; c < channels_per_group; ++c) {
                    for (int s = 0; s < spatial_size; ++s) {
                        int idx = b * (num_groups * channels_per_group * spatial_size) +
                                  group_channels_start * spatial_size + c * spatial_size + s;
                        sum += x[idx];
                    }
                }
            }

            sum = block_sum(sum, buffer);
            const scalar_t mean = sum / (batch_size * channels_per_group * spatial_size);

            scalar_t var_sum = 0.0;
            for (int b = 0; b < batch_size; ++b) {
                for (int c = 0; c < channels_per_group; ++c) {
                    for (int s = 0; s < spatial_size; ++s) {
                        int idx = b * (num_groups * channels_per_group * spatial_size) +
                                  group_channels_start * spatial_size + c * spatial_size + s;
                        scalar_t x_centered = x[idx] - mean;
                        var_sum += x_centered * x_centered;
                    }
                }
            }
            var_sum = block_sum(var_sum, buffer);
            const scalar_t var = var_sum / (batch_size * channels_per_group * spatial_size);
            const scalar_t std = sqrt(var + eps);

            // Normalize and apply affine
            scalar_t normalized = (x_val - mean) / std;
            output[x_index] = weight[channel] * normalized + bias[channel];
        }

        // Helper function for block-wise summation
        template <typename scalar_t>
        __device__ scalar_t block_sum(scalar_t value, scalar_t* smem) {
            __syncthreads();
            int tid = threadIdx.x;
            smem[tid] = value;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    smem[tid] += smem[tid + s];
                }
                __syncthreads();
            }
            return smem[0];
        }

        torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                     int num_groups, float eps) {
            const int batch_size = x.size(0);
            const int channels = x.size(1);
            const int spatial_size = x.numel() / (batch_size * channels);
            const int channels_per_group = channels / num_groups;

            auto output = torch::empty_like(x);

            dim3 threads(1, 1, 1, 1);  // Adjust based on dimensions
            dim3 blocks(1, 1, num_groups, 1);

            // Define grid and block dimensions appropriately
            // This example uses a simplified launch configuration; tune for performance
            auto stream = at::cuda::getCurrentCUDAStream();
            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_cuda", ([&] {
                group_norm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, num_groups, channels_per_group, spatial_size, eps);
            }));

            return output;
        }
        """

        group_norm_cpp_source = """
        torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);
        """

        self.group_norm = load_inline(
            name="group_norm",
            cpp_sources=group_norm_cpp_source,
            cuda_sources=group_norm_source,
            functions=["group_norm_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm.group_norm_cuda(
            x.cuda(), self.weight.cuda(), self.bias.cuda(),
            self.num_groups, self.eps
        )

def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features, 8]