import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_forward_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        y[idx] = xi / (1.0f + expf(-xi));
    }
}

__global__ void swish_backward_kernel(const float* x, const float* dy, float* dx, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sigmoid_xi = 1.0f / (1.0f + expf(-xi));
        dx[idx] = dy[idx] * (sigmoid_xi + xi * (1.0f - sigmoid_xi));
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    const int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    return y;
}

torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor dy) {
    auto dx = torch::empty_like(x);
    const int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), dy.data_ptr<float>(), dx.data_ptr<float>(), size);
    return dx;
}

class SwishFunction : public torch::autograd::Function<SwishFunction> {
    static torch::Tensor forward(torch::AutogradContext* ctx, torch::Tensor x) {
        ctx->save_for_backward({x});
        return swish_forward_cuda(x);
    }
    static torch::Tensor backward(torch::AutogradContext* ctx, torch::Tensor dy) {
        auto saved = ctx->get_saved_tensors();
        auto x = saved[0];
        return swish_backward_cuda(x, dy);
    }
};

torch::Tensor swish(torch::Tensor x) {
    return SwishFunction::apply(x);
}
"""

swish_cpp_header = """
torch::Tensor swish_forward_cuda(torch::Tensor x);
torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor dy);
torch::Tensor swish(torch::Tensor x);
"""

swish_cuda = load_inline(
    name="swish_cuda",
    cpp_sources=swish_cpp_header,
    cuda_sources=swish_source,
    functions=["swish"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = swish_cuda.swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(x)