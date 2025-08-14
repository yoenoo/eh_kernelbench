import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for instance normalization
inorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void instance_norm_forward(const scalar_t* x, scalar_t* y, const float* gamma, const float* beta,
                                      int batch_size, int channels, int height, int width) {
    CUDA_KERNEL_LOOP(index, batch_size * channels * height * width) {
        const int w = index % width;
        const int h = (index / width) % height;
        const int c = (index / (width * height)) % channels;
        const int n = index / (channels * height * width);

        // Compute mean and variance for the current instance (n, c)
        float mean = 0.0;
        float variance = 0.0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int pos = n * channels * height * width + c * height * width + i * width + j;
                mean += static_cast<float>(x[pos]);
            }
        }
        mean /= (height * width);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int pos = n * channels * height * width + c * height * width + i * width + j;
                variance += pow(static_cast<float>(x[pos]) - mean, 2);
            }
        }
        variance /= (height * width);

        float std = sqrt(variance + 1e-5);
        float inv_std = 1.0 / std;

        // Normalize and scale/shift
        int pos = n * channels * height * width + c * height * width + h * width + w;
        y[pos] = (static_cast<float>(x[pos]) - mean) * inv_std * gamma[c] + beta[c];
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                                int batch_size, int channels, int height, int width) {
    auto output = torch::empty_like(x);

    const int num_threads = 512;
    const int num_elements = x.numel();
    const int num_blocks = (num_elements + num_threads - 1) / num_threads;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "instance_norm_cuda", ([&] {
        instance_norm_forward<scalar_t><<<num_blocks, num_threads>>>(
            x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            batch_size, channels, height, width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

inorm_cpp_source = "torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int batch_size, int channels, int height, int width);"

# Compile the custom CUDA kernel
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=inorm_cpp_source,
    cuda_sources=inorm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.custom_inorm = instance_norm

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        output = self.custom_inorm.instance_norm_cuda(
            x.cuda(),
            self.gamma.cuda(),
            self.beta.cuda(),
            batch_size,
            channels,
            height,
            width
        )
        return output