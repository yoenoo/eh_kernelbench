#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val < -1.0f) {
            out[idx] = -1.0f;
        } else if (val > 1.0f) {
            out[idx] = 1.0f;
        } else {
            out[idx] = val;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}