import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

        # Define and load custom CUDA kernel for group normalization
        group_norm_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void group_norm_kernel(
            scalar_t* __restrict__ x,
            scalar_t* __restrict__ weight,
            scalar_t* __restrict__ bias,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int num_groups,
            const int channels_per_group,
            const int spatial_size,
            const float eps
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int group_id = blockIdx.y;

            if (idx >= channels_per_group * spatial_size) return;

            int c_offset = group_id * channels_per_group;
            int offset = c_offset * spatial_size * batch_size;

            // Compute mean and variance for this group
            float mean = 0;
            float var = 0;
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < channels_per_group; ++c) {
                    for (int s = 0; s < spatial_size; ++s) {
                        int pos = offset + (n * channels_per_group + c) * spatial_size + s;
                        mean += x[pos];
                    }
                }
            }
            mean /= (batch_size * channels_per_group * spatial_size);

            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < channels_per_group; ++c) {
                    for (int s = 0; s < spatial_size; ++s) {
                        int pos = offset + (n * channels_per_group + c) * spatial_size + s;
                        float val = x[pos] - mean;
                        var += val * val;
                    }
                }
            }
            var /= (batch_size * channels_per_group * spatial_size);
            float std = 1.0 / sqrt(var + eps);

            // Normalize and apply affine
            for (int tid = threadIdx.x; tid < channels_per_group * spatial_size; tid += blockDim.x) {
                int c = tid / spatial_size;
                int s = tid % spatial_size;
                for (int n = 0; n < batch_size; ++n) {
                    int pos = offset + (n * channels_per_group + c) * spatial_size + s;
                    output[pos] = (x[pos] - mean) * std * weight[c_offset + c] + bias[c_offset + c];
                }
            }
        }

        torch::Tensor group_norm_cuda(
            torch::Tensor x,
            torch::Tensor weight,
            torch::Tensor bias,
            int num_groups,
            float eps
        ) {
            const int batch_size = x.size(0);
            const int channels = x.size(1);
            const int spatial_size = x.size(2) * x.size(3);
            const int channels_per_group = channels / num_groups;

            auto output = torch::empty_like(x);

            dim3 threads(256);
            dim3 blocks(
                (channels_per_group * spatial_size + threads.x - 1) / threads.x,
                num_groups
            );

            // Launch kernel
            AT_DISPATCH_FLOATING_TYPES(x.type(), "group_norm_cuda", ([&] {
                group_norm_kernel<scalar_t><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    num_groups,
                    channels_per_group,
                    spatial_size,
                    eps
                );
            }));

            return output;
        }
        """

        group_norm_cpp_source = (
            "torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);"
        )

        self.group_norm = load_inline(
            name="group_norm",
            cpp_sources=group_norm_cpp_source,
            cuda_sources=group_norm_source,
            functions=["group_norm_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manually perform group normalization using CUDA kernel
        return self.group_norm.group_norm_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.bias.cuda(),
            self.num_groups,
            self.eps
        )

def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []