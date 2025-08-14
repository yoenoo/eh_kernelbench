import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused triplet margin loss
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void triplet_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    float margin,
    float* loss_output,
    int batch_size,
    int dim
) {
    extern __shared__ scalar_t shared_loss[];
    int tid = threadIdx.x;
    int bid = blockIdx.x * blockDim.x + tid;

    shared_loss[tid] = 0.0;

    if (bid < batch_size) {
        scalar_t ap_dist_sq = 0.0;
        scalar_t an_dist_sq = 0.0;
        for (int d = 0; d < dim; ++d) {
            scalar_t ap_diff = anchor[bid * dim + d] - positive[bid * dim + d];
            ap_dist_sq += ap_diff * ap_diff;
            scalar_t an_diff = anchor[bid * dim + d] - negative[bid * dim + d];
            an_dist_sq += an_diff * an_diff;
        }

        scalar_t ap_dist = sqrt(ap_dist_sq);
        scalar_t an_dist = sqrt(an_dist_sq);
        scalar_t loss_val = ap_dist - an_dist + margin;
        if (loss_val > 0) {
            shared_loss[tid] = loss_val;
        } else {
            shared_loss[tid] = 0.0;
        }
    } else {
        shared_loss[tid] = 0.0;
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_output, shared_loss[0]);
    }
}

torch::Tensor triplet_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    const auto batch_size = anchor.size(0);
    const auto dim = anchor.size(1);
    auto loss = torch::zeros(1, device=anchor.device());

    const int block_size = 256;
    dim3 blocks((batch_size + block_size - 1) / block_size);
    dim3 threads(block_size);
    size_t shared_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_loss_cuda", ([&] {
        triplet_loss_kernel<scalar_t><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
            anchor.data<scalar_t>(),
            positive.data<scalar_t>(),
            negative.data<scalar_t>(),
            margin,
            loss.data<float>(),
            batch_size,
            dim
        );
    }));

    return loss / batch_size; // Average the loss
}
"""

triplet_loss_cpp_source = (
    "torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"
)

# Compile the inline CUDA code for fused triplet loss
triplet_loss = load_inline(
    name="triplet_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_loss_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss = triplet_loss

    def forward(self, anchor, positive, negative):
        return self.triplet_loss.triplet_loss_cuda(anchor, positive, negative, self.margin)