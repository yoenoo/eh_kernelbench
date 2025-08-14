import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void kl_div_forward_kernel(const scalar_t* log_probs, const scalar_t* targets, scalar_t* output, int batch_size, int dim_size) {
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    if (element_idx >= dim_size) return;

    scalar_t term = targets[batch_idx * dim_size + element_idx] * 
                   (log((targets[batch_idx * dim_size + element_idx] + 1e-8)) - 
                    log_probs[batch_idx * dim_size + element_idx]);
    atomicAdd(output, term);
}

at::Tensor kl_div_forward_cuda(at::Tensor log_probs, at::Tensor targets, const std::string& reduction) {
    const int batch_size = log_probs.size(0);
    const int dim_size = log_probs.size(1);

    auto output = at::zeros({batch_size}, log_probs.type());

    const int block_size = 1024;
    dim3 grid(batch_size);
    dim3 block(dim_size);

    AT_DISPATCH_FLOATING_TYPES(log_probs.type(), "kl_div_forward_cuda", ([&] {
        kl_div_forward_kernel<scalar_t><<<grid, block>>>(
            log_probs.data<scalar_t>(), targets.data<scalar_t>(), 
            output.data<scalar_t>(), batch_size, dim_size);
    }));

    if (reduction == "batchmean") {
        output = output.sum() / batch_size;
    }
    return output;
}
"""

kl_div_cpp_source = """
at::Tensor kl_div_forward_cuda(at::Tensor log_probs, at::Tensor targets, const std::string& reduction);
"""

kl_div = load_inline(
    name="kl_div_forward",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div_forward = kl_div

    def forward(self, predictions, targets):
        log_probs = torch.log(predictions)
        return self.kl_div_forward.kl_div_forward_cuda(log_probs, targets, "batchmean")