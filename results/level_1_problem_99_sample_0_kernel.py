import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void triplet_loss_kernel(const scalar_t* anchor, const scalar_t* positive,
                                   const scalar_t* negative, scalar_t* loss_values,
                                   const scalar_t margin, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        scalar_t ap_dist = 0.0;
        scalar_t an_dist = 0.0;
        for (int d = 0; d < dim; ++d) {
            scalar_t ap_diff = anchor[idx * dim + d] - positive[idx * dim + d];
            scalar_t an_diff = anchor[idx * dim + d] - negative[idx * dim + d];
            ap_dist += ap_diff * ap_diff;
            an_dist += an_diff * an_diff;
        }
        ap_dist = sqrt(ap_dist);
        an_dist = sqrt(an_dist);
        scalar_t loss = ap_dist - an_dist + margin;
        loss_values[idx] = (loss > 0) ? loss : 0;
    }
}

torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive,
                               torch::Tensor negative, float margin) {
    const int batch_size = anchor.size(0);
    const int dim = anchor.size(1);
    auto loss = torch::zeros({batch_size}, torch::device("cuda"));

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(anchor.type(), "triplet_loss_cuda", ([&] {
        triplet_loss_kernel<scalar_t><<<num_blocks, block_size>>>(
            anchor.data_ptr<scalar_t>(), positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(), loss.data_ptr<scalar_t>(),
            margin, batch_size, dim);
    }));

    return loss.mean();
}
"""

triplet_loss_cpp_source = """
torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive,
                               torch::Tensor negative, float margin);
"""

triplet_loss_cuda = load_inline(
    name="triplet_loss_cuda",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_loss_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss_cuda = triplet_loss_cuda

    def forward(self, anchor, positive, negative):
        return self.triplet_loss_cuda.triplet_loss_cuda(
            anchor, positive, negative, self.margin
        )