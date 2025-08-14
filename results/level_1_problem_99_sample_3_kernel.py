import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Triplet Margin Loss
triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* anchor,
    const scalar_t* positive,
    const scalar_t* negative,
    scalar_t* loss,
    const float margin,
    int batch_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        scalar_t ap_dist = 0.0;
        scalar_t an_dist = 0.0;
        for (int d = 0; d < dim; ++d) {
            ap_dist += (anchor[idx * dim + d] - positive[idx * dim + d]) * (anchor[idx * dim + d] - positive[idx * dim + d]);
            an_dist += (anchor[idx * dim + d] - negative[idx * dim + d]) * (anchor[idx * dim + d] - negative[idx * dim + d]);
        }
        ap_dist = sqrt(ap_dist);
        an_dist = sqrt(an_dist);
        loss[idx] = (ap_dist - an_dist + margin) > 0.0 ? ap_dist - an_dist + margin : 0.0;
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    auto batch_size = anchor.size(0);
    auto dim = anchor.size(1);
    auto loss = torch::zeros({batch_size}, anchor.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(anchor.type(), "triplet_margin_loss_cuda", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<num_blocks, block_size>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            loss.data_ptr<scalar_t>(),
            margin,
            batch_size,
            dim
        );
    }));

    return loss.mean();
}
"""

triplet_margin_loss_cpp_source = "torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"

# Compile the inline CUDA code
triplet_margin_loss = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss = triplet_margin_loss  # Reference to the CUDA kernel

    def forward(self, anchor, positive, negative):
        return self.triplet_loss.triplet_margin_loss_cuda(anchor, positive, negative, self.margin)

def get_inputs():
    batch_size = 32768
    input_shape = (8192,)
    dim = 1
    scale = torch.rand(())
    return [
        torch.rand(batch_size, *input_shape, dtype=torch.float32).cuda() * scale,
        torch.rand(batch_size, *input_shape, dtype=torch.float32).cuda(),
        torch.rand(batch_size, *input_shape, dtype=torch.float32).cuda()
    ]

def get_init_inputs():
    return [1.0]  # Default margin