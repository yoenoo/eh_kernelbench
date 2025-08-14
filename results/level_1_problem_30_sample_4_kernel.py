import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

        # Define and compile the custom CUDA kernel for Softsign
        softsign_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void softsign_kernel(const float* x, float* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float xi = x[idx];
                out[idx] = xi / (1.0f + fabs(xi));
            }
        }

        torch::Tensor softsign_cuda(torch::Tensor x) {
            auto size = x.numel();
            auto out = torch::empty_like(x);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            softsign_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

            return out;
        }
        """

        softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor x);"

        # Load the inline CUDA extension
        self.softsign = load_inline(
            name="softsign",
            cpp_sources=softsign_cpp_source,
            cuda_sources=softsign_source,
            functions=["softsign_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom CUDA kernel for the Softsign operation
        return self.softsign.softsign_cuda(x)