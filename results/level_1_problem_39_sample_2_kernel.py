import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Compile the CUDA kernel here
        l2norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void l2norm_kernel(const float* input, float* output, const int batch_size, const int dim) {
            int batch_idx = blockIdx.x;
            int element_idx = threadIdx.x;
            if (element_idx >= dim) return;

            float val = input[batch_idx * dim + element_idx];
            float sum_sq = 0.0;
            for (int i = 0; i < dim; ++i) {
                sum_sq += input[batch_idx * dim + i] * input[batch_idx * dim + i];
            }
            float inv_norm = 1.0f / sqrtf(sum_sq + 1e-12f);
            output[batch_idx * dim + element_idx] = val * inv_norm;
        }

        torch::Tensor l2norm_cuda(torch::Tensor input) {
            int batch_size = input.size(0);
            int dim = input.size(1);
            auto output = torch::empty_like(input);

            dim3 threads(dim);
            dim3 blocks(batch_size);

            l2norm_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

            cudaDeviceSynchronize();
            return output;
        }
        """
        l2norm_cpp_source = "torch::Tensor l2norm_cuda(torch::Tensor input);"

        self.l2norm = load_inline(
            name="l2norm",
            cpp_sources=l2norm_cpp_source,
            cuda_sources=l2norm_source,
            functions=["l2norm_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.l2norm.l2norm_cuda(x)