import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define and compile the custom CUDA kernel for L1 normalization
        l1_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void l1_norm_kernel(const float* x_data, float* out_data, int batch_size, int dim) {
            int batch = blockIdx.x;
            int index = threadIdx.x;

            __shared__ float sum[256]; // Assuming dim <= 65535, so 256 threads per block can accumulate via shared memory
            sum[index] = 0.0;

            for (int i = index; i < dim; i += blockDim.x) {
                float val = abs(x_data[batch * dim + i]);
                sum[index] += val;
            }

            __syncthreads();

            // Sequential reduction in shared memory
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (index < stride) {
                    sum[index] += sum[index + stride];
                }
                __syncthreads();
            }

            if (index == 0) {
                float mean_val = sum[0] / dim;
                for (int i = 0; i < dim; i++) {
                    float val = x_data[batch * dim + i];
                    out_data[batch * dim + i] = val / mean_val;
                }
            }
        }

        torch::Tensor l1_norm_cuda(torch::Tensor x) {
            int batch_size = x.size(0);
            int dim = x.size(1);

            auto out = torch::empty_like(x);
            int block_size = dim; // Each block handles one batch sample
            dim3 threads(dim < 512 ? dim : 512); // Ensure threads <= 1024 (max per block)
            dim3 blocks(batch_size);

            l1_norm_kernel<<<blocks, threads>>>(
                x.data_ptr<float>(),
                out.data_ptr<float>(),
                batch_size,
                dim
            );

            return out;
        }
        """

        # Inline compilation of the CUDA kernel
        l1_norm_cpp_source = "torch::Tensor l1_norm_cuda(torch::Tensor x);"
        self.l1_norm_cuda = load_inline(
            name="l1_norm_cuda",
            cpp_sources=l1_norm_cpp_source,
            cuda_sources=l1_norm_source,
            functions=["l1_norm_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm_cuda.l1_norm_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []