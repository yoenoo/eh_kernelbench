import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inline CUDA kernel for optimized KL Divergence calculation
        kl_div_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename T>
        __global__ void kl_div_kernel(const T* log_probs, const T* targets, T* output, 
                                     int batch_size, int dim_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                T sum = 0;
                for (int d = 0; d < dim_size; ++d) {
                    T p = targets[idx * dim_size + d];
                    T q = log_probs[idx * dim_size + d];
                    if (p > 1e-20) {  // Avoid log(0) and p*ln(0)
                        sum += p * (torch::log(p) - q);
                    }
                }
                output[idx] = sum / dim_size;
            }
        }

        torch::Tensor kl_div_cuda(torch::Tensor log_probs, torch::Tensor targets) {
            const int batch_size = log_probs.size(0);
            const int dim_size = log_probs.size(1);

            auto output = torch::empty({batch_size}, log_probs.options());

            const int threads = 256;
            const int blocks = (batch_size + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(log_probs.type(), "kl_div_cuda", ([&] {
                kl_div_kernel<scalar_t><<<blocks, threads>>>(
                    log_probs.data<scalar_t>(), 
                    targets.data<scalar_t>(), 
                    output.data<scalar_t>(),
                    batch_size, 
                    dim_size);
            }));

            return output.mean();
        }
        """

        kl_div_header = "torch::Tensor kl_div_cuda(torch::Tensor log_probs, torch::Tensor targets);"

        # Compile the custom kernel
        self.kl_div = load_inline(
            name="kl_div_cuda",
            cpp Sources=kl_div_header,
            cuda_sources=kl_div_source,
            functions=["kl_div_cuda"],
            verbose=True
        )

    def forward(self, predictions, targets):
        # Check inputs are probabilities
        assert predictions.requires_grad, "Predictions must require gradient"
        log_probs = predictions.log()  # Moved log inside to ensure contiguous memory
        return self.kl_div.kl_div_cuda(log_probs, targets)

# Maintain the original input generation functions
batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), 
            torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []