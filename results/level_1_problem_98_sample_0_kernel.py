import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for KL Divergence computation
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void kl_div_forward_kernel(const scalar_t* predictions, const scalar_t* targets, scalar_t* output, int batch_size, int dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim_size) {
        int batch_idx = idx / dim_size;
        int dim_idx = idx % dim_size;
        scalar_t log_p = __logf(predictions[idx]);
        output[batch_idx] += (log_p - log(targets[idx])) * predictions[idx];
    }
}

at::Tensor kl_div_forward_cuda(at::Tensor predictions, at::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int dim_size = predictions.size(1);
    auto output = at::zeros({batch_size}, predictions.type());

    const int total_elements = batch_size * dim_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(predictions.type(), "kl_div_forward_cuda", ([&] {
        kl_div_forward_kernel<scalar_t><<<num_blocks, block_size>>>(
            predictions.data<scalar_t>(), targets.data<scalar_t>(), 
            output.data<scalar_t>(), batch_size, dim_size);
    }));

    output /= dim_size;  // Since reduction is 'batchmean', divide by dim_size (equivalent to averaging over classes)
    return output;
}
"""

kl_div_forward_cpp_source = (
    "at::Tensor kl_div_forward_cuda(at::Tensor predictions, at::Tensor targets);"
)

# Compile the inline CUDA code for KL Divergence
kl_div_forward = load_inline(
    name="kl_div_forward",
    cpp_sources=kl_div_forward_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div_forward = kl_div_forward

    def forward(self, predictions, targets):
        return self.kl_div_forward.kl_div_forward_cuda(predictions, targets)

def get_inputs():
    batch_size = 8192 * 2
    input_shape = (8192 * 2,)
    dim = 1
    scale = torch.rand(())
    return [
        (torch.rand(batch_size, *input_shape) * scale).softmax(dim=-1),
        torch.rand(batch_size, *input_shape).softmax(dim=-1),
    ]

def get_init_inputs():
    return []