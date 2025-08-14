import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Inline CUDA kernel for Hinge Loss computation fused into a single kernel
        hinge_loss_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* output, int batch_size) {
            extern __shared__ volatile float shared[];
            unsigned int tid = threadIdx.x;
            unsigned int idx = blockIdx.x * blockDim.x + tid;
            float loss = 0.0;
            if (idx < batch_size) {
                float val = 1.0 - predictions[idx] * targets[idx];
                loss = fmaxf(val, 0.0);
            }
            // Use shared memory to compute the mean
            shared[tid] = (idx < batch_size) ? loss : 0.0;
            __syncthreads();
            
            // Bitwise reduction in shared memory
            for (int s=blockDim.x/2; s>0; s>>=1) {
                if (tid < s) {
                    shared[tid] += shared[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                atomicAdd(output, shared[0]);
            }
        }

        torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
            const int batch_size = predictions.size(0);
            const int block_size = 256;
            const int num_blocks = (batch_size + block_size - 1) / block_size;

            auto output = torch::zeros(1, predictions.device());
            
            dim3 blocks(num_blocks);
            dim3 threads(block_size);
            const unsigned int smem_size = threads.x * sizeof(float);
            hinge_loss_kernel<<<blocks, threads, smem_size, at::cuda::getCurrentCUDAStream()>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), batch_size);

            // Final division to compute the mean
            output.div_(batch_size);
            return output;
        }
        """

        hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

        # Load the CUDA extension
        self.hinge_loss = load_inline(
            name="hinge_loss",
            cpp_sources=[hinge_loss_cpp_source],
            cuda_sources=[hinge_loss_source],
            functions=["hinge_loss_cuda"],
            verbose=True
        )

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)