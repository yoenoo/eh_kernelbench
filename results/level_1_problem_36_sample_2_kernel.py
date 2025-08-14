import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Define and load the custom CUDA kernel for RMS normalization
        rms_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        template <typename scalar_t>
        __global__ void rms_norm_forward_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, const int batch_size, const int features, const int dim_product, const float eps) {
            int batch_idx = blockIdx.x;
            int element_idx = threadIdx.x;

            __shared__ scalar_t shared_features[512]; // size needs to be at least features

            // Load data into shared memory
            for (int i = element_idx; i < features; i += blockDim.x) {
                shared_features[i] = x[batch_idx * features * dim_product + i * dim_product + element_idx];
            }
            __syncthreads();

            // Compute mean of squares
            scalar_t sum = 0.0;
            for (int i = 0; i < features; i++) {
                sum += shared_features[i] * shared_features[i];
            }
            sum /= features;
            scalar_t rms = rsqrt(sum + eps);

            __syncthreads();

            // Normalize and write output
            for (int i = element_idx; i < features * dim_product; i += blockDim.x) {
                out[batch_idx * features * dim_product + i] = x[batch_idx * features * dim_product + i] * rms;
            }
        }

        torch::Tensor rms_norm_forward(torch::Tensor x, float eps) {
            const int batch_size = x.size(0);
            const int features = x.size(1);
            const int dim_product = x.numel() / (batch_size * features);

            auto out = torch::empty_like(x);
            const int threads = 512;
            const dim3 blocks(batch_size);
            const dim3 threadsPerBlock(threads);

            AT_DISPATCH_FLOATING_TYPES(x.type(), "rms_norm_forward", ([&] {
                rms_norm_forward_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
                    x.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    batch_size,
                    features,
                    dim_product,
                    eps
                );
            }));

            return out;
        }
        """

        rms_norm_cpp_source = """
        torch::Tensor rms_norm_forward(torch::Tensor x, float eps);
        """

        self.rms_norm = load_inline(
            name="rms_norm",
            cpp_sources=rms_norm_cpp_source,
            cuda_sources=rms_norm_source,
            functions=["rms_norm_forward"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm.rms_norm_forward(x, self.eps)

def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x.cuda()]

def get_init_inputs():
    return []