import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void SoftmaxForwardKernel(const scalar_t* __restrict__ input,
                                    scalar_t* __restrict__ output,
                                    int batch_size,
                                    int dim) {
    extern __shared__ scalar_t shared_mem[];

    int row = blockIdx.x;
    int col = threadIdx.x;

    scalar_t val = input[row * dim + col];
    scalar_t exp_val = exp(val);

    // Store in shared memory
    shared_mem[col] = exp_val;
    __syncthreads();

    // Block-wise reduction to compute the sum
    for (int stride = dim >> 1; stride > 0; stride >>= 1) {
        if (col < stride) {
            shared_mem[col] += shared_mem[col + stride];
        }
        __syncthreads();
    }

    scalar_t sum = (col == 0) ? shared_mem[0] : 0.0;
    __syncthreads();

    // Broadcast sum and compute output
    scalar_t softmax_val = exp_val / sum;
    output[row * dim + col] = softmax_val;
}

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);

    torch::Tensor output = torch::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(dim);
    size_t shared_mem = threads.x * sizeof(float);

    SoftmaxForwardKernel<float>
        <<<blocks, threads, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            dim);

    return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_forward_cuda(torch::Tensor input);"

softmax_ops = load_inline(
    name="softmax_ops",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions="softmax_forward_cuda",
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_forward_cuda = softmax_ops

    def forward(self, x):
        return self.softmax_forward_cuda.softmax_forward_cuda(x)