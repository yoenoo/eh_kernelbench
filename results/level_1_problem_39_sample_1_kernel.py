import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l2_norm_kernel(float* x, float* output, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Compute the norm for the current row
        float norm = 0.0;
        for (int d = 0; d < dim; ++d) {
            float val = x[idx * dim + d];
            norm += val * val;
        }
        norm = sqrtf(norm);
        if (norm == 0.0) norm = 1.0e-12; // Avoid division by zero

        // Normalize the elements
        for (int d = 0; d < dim; ++d) {
            output[idx * dim + d] = x[idx * dim + d] / norm;
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int batch_size = x.size(0);
    int dim = x.size(1);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    l2_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

l2norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code for L2 normalization
l2norm = load_inline(
    name="l2norm",
    cpp_sources=l2norm_cpp_source,
    cuda_sources=l2norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2norm_cuda = l2norm

    def forward(self, x):
        return self.l2norm_cuda.l2_norm_cuda(x)