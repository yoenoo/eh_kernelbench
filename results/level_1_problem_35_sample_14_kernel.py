import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

groupnorm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

template <typename scalar_t>
__global__ void groupnorm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const float* gamma,
    const float* beta,
    const int batch_size,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps) {

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int group = blockIdx.y;

    if (group >= num_groups) return;

    const int group_channels_start = group * channels_per_group;
    const int spatial_offset = blockIdx.z * spatial_size;

    extern __shared__ scalar_t shared_mem[];
    scalar_t* sum_buf = shared_mem;
    scalar_t* sum_sq_buf = sum_buf + blockDim.x;

    scalar_t sum = 0.0;
    scalar_t sum_sq = 0.0;

    // Iterate over channels in the group
    for (int c = gid; c < channels_per_group; c += blockDim.x) {
        const int in_channel = group_channels_start + c;
        const int elem_idx = in_channel * spatial_size + spatial_offset;
        scalar_t val = input[elem_idx];
        sum += val;
        sum_sq += val * val;
    }

    __syncwarp();

    // Block-wise reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, stride);
    }

    if (threadIdx.x == 0) {
        sum_buf[0] = sum;
        sum_sq_buf[0] = sum_sq;
    }
    __syncthreads();

    scalar_t mean = sum_buf[0] / (channels_per_group * spatial_size);
    scalar_t var = sum_sq_buf[0] / (channels_per_group * spatial_size) - mean * mean;
    scalar_t std_inv = 1.0f / sqrtf(var + eps);

    // Write back normalized values
    for (int c = gid; c < channels_per_group; c += blockDim.x) {
        const int in_channel = group_channels_start + c;
        const int elem_idx = in_channel * spatial_size + spatial_offset;
        scalar_t val = input[elem_idx];
        scalar_t normalized = (val - mean) * std_inv;
        output[elem_idx] = normalized * gamma[c] + beta[c];
    }
}

torch::Tensor groupnorm_cuda(torch::Tensor input,
                             torch::Tensor gamma,
                             torch::Tensor beta,
                             int num_groups,
                             float eps = 1e-5) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int spatial_size = input.numel() / (batch_size * channels);
    const int channels_per_group = channels / num_groups;

    auto output = torch::empty_like(input);

    dim3 threads(256);
    dim3 blocks((channels_per_group + threads.x - 1)/threads.x,
                num_groups,
                spatial_size);

    const int shared_size = 2 * sizeof(float) * threads.x;
    groupnorm_kernel<float><<<blocks, threads.x, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        num_groups,
        channels_per_group,
        spatial_size,
        eps
    );

    return output;
}
"""

groupnorm_cpp_source = """
torch::Tensor groupnorm_cuda(torch::Tensor input,
                            torch::Tensor gamma,
                            torch::Tensor beta,
                            int num_groups,
                            float eps = 1e-5);
"""

groupnorm_cuda = load_inline(
    name="groupnorm_cuda",
    cpp_sources=groupnorm_cpp_source,
    cuda_sources=groupnorm_source,
    functions=["groupnorm_cuda"],
    verbose=True,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.groupnorm_cuda = groupnorm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.groupnorm_cuda.groupnorm_cuda(
            x.cuda(),
            self.gamma.cuda(),
            self.beta.cuda(),
            self.num_groups,
            self.eps
        )