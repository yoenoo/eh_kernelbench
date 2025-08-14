import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

__global__ void instance_norm_kernel(const float* input, float* output, 
                                    const float* weight, const float* bias,
                                    int batch, int features, int h, int w,
                                    float eps) {
    extern __shared__ float stats[];
    int feat_step = h * w;
    int item = blockIdx.x;
    int feature = item % features;
    item /= features;
    int n = item;

    int idx = n * features * feat_step + feature * feat_step;
    float* data = (float*)&input[idx];

    // Compute mean and variance in shared memory
    float mean = 0, var = 0;
    #pragma unroll
    for (int i = threadIdx.x; i < feat_step; i += blockDim.x) {
        stats[threadIdx.x] = data[i];
    }
    __shared__ float local_mean, local_var;
    if (threadIdx.x == 0) {
        mean = var = 0;
        for (int i = 0; i < feat_step; i++) {
            mean += stats[i];
        }
        mean /= feat_step;

        for (int i = 0; i < feat_step; i++) {
            var += (stats[i] - mean) * (stats[i] - mean);
        }
        var = var / feat_step + eps;
        local_mean = mean;
        local_var = rsqrtf(var);
    }
    __syncthreads();

    if (threadIdx.x < feat_step) {
        float normalized = (stats[threadIdx.x] - local_mean) * local_var;
        if (weight) normalized *= weight[feature];
        if (bias) normalized += bias[feature];
        output[idx + threadIdx.x] = normalized;
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor input, 
                                torch::Tensor weight,
                                torch::Tensor bias,
                                float eps) {
    const int batch = input.size(0);
    const int features = input.size(1);
    const int h = input.size(2);
    const int w = input.size(3);
    const int elements = h * w;
    const int total_items = batch * features;

    auto output = torch::empty_like(input);

    dim3 blocks(total_items);
    dim3 threads(std::min(elements, 1024));
    int shared_size = elements * sizeof(float);

    instance_norm_kernel<<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch, features, h, w,
        eps
    );

    return output;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float eps);"
)

instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
    extra_cflags=["-DCUDA_HAS_FP16=1", "-D__CUDA_NO_HALF_OPERATORS__"],
    extra_cuda_cflags=["-lineinfo", "-G"]
)

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = 1e-5
        self.instance_norm = instance_norm

    def forward(self, x):
        return self.instance_norm.instance_norm_cuda(x, self.weight, self.bias, self.eps)