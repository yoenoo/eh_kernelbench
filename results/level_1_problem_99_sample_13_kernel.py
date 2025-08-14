import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        # Compile the custom CUDA kernel for Triplet Margin Loss
        self.triplet_margin_loss_cuda = self._load_triplet_margin_loss_cuda()
    
    def _load_triplet_margin_loss_cuda(self):
        triplet_margin_loss_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <cmath>

        template <typename scalar_t>
        __global__ void triplet_margin_loss_kernel(
            const scalar_t* __restrict__ anchor,
            const scalar_t* __restrict__ positive,
            const scalar_t* __restrict__ negative,
            scalar_t* __restrict__ output,
            scalar_t margin,
            int batch_size,
            int dim) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size) return;

            scalar_t dist_pos = 0.0;
            scalar_t dist_neg = 0.0;
            for (int d = 0; d < dim; ++d) {
                scalar_t a = anchor[idx * dim + d] - positive[idx * dim + d];
                scalar_t b = anchor[idx * dim + d] - negative[idx * dim + d];
                dist_pos += a * a;
                dist_neg += b * b;
            }
            dist_pos = sqrt(dist_pos);
            dist_neg = sqrt(dist_neg);
            scalar_t loss_val = dist_pos - dist_neg + margin;
            output[idx] = (loss_val > 0) ? loss_val : 0.0;
        }

        torch::Tensor triplet_margin_loss_cuda(
            torch::Tensor anchor,
            torch::Tensor positive,
            torch::Tensor negative,
            float margin,
            int batch_size,
            int dim) {

            const int threads = 256;
            const int blocks = (batch_size + threads - 1) / threads;

            auto output = torch::zeros({batch_size}, anchor.options());

            AT_DISPATCH_FLOATING_TYPES(anchor.type(), "triplet_margin_loss_cuda", ([&] {
                triplet_margin_loss_kernel<scalar_t><<<blocks, threads>>>(
                    anchor.data_ptr<scalar_t>(),
                    positive.data_ptr<scalar_t>(),
                    negative.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    margin,
                    batch_size,
                    dim);
            }));

            return output.mean();
        }

        """
        
        header = """
        torch::Tensor triplet_margin_loss_cuda(
            torch::Tensor anchor,
            torch::Tensor positive,
            torch::Tensor negative,
            float margin,
            int batch_size,
            int dim);
        """
        module = load_inline(
            name="triplet_margin_loss_cuda",
            cpp_sources=header,
            cuda_sources=triplet_margin_source,
            functions=["triplet_margin_loss_cuda"],
            verbose=True
        )
        return module.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        batch_size = anchor.size(0)
        dim = torch.tensor(anchor.shape[1:])
        return self.triplet_margin_loss_cuda(anchor, positive, negative, self.margin, batch_size, dim.item())