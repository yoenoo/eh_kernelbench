import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the custom CUDA kernel for Hinge Loss computation
        hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out, int batch_size, int dim) {
    // Each thread processes one element of the batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float pred = predictions[idx];
        float target = targets[idx];
        float val = 1.0f - pred * target;
        out[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto out = torch::empty(batch_size, predictions.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        1  // dim is fixed to 1 in the original model
    );

    // Compute mean
    auto sum = torch::sum(out);
    return sum / batch_size;
}
"""

        hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

        # Compile the kernel
        self.hinge_loss_op = load_inline(
            name="hinge_loss",
            cpp_sources=hinge_loss_cpp_source,
            cuda_sources=hinge_loss_source,
            functions=["hinge_loss_cuda"],
            verbose=True
        )

    def forward(self, predictions, targets):
        # Ensure targets are in {-1, 1}
        targets = targets.float() * 2.0 - 1.0
        return self.hinge_loss_op.hinge_loss_cuda(predictions.view(-1), targets.view(-1))

def get_inputs():
    return [
        torch.rand(batch_size, *input_shape).cuda(),
        torch.randint(0, 2, (batch_size,)).float().cuda()
    ]

def get_init_inputs():
    return []