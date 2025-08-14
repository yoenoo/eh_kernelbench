#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void swish_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float exp_val = expf(-x[idx]);
        float sigmoid_x = 1.0f / (1.0f + exp_val);
        y[idx] = x[idx] * sigmoid_x;
    }
}