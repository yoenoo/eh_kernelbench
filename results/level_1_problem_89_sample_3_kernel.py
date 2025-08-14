import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Load the custom CUDA kernel for the prefix sum operation
        prefix_sum_kernel = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <ATen/ATen.h>
        #include <ATen/cuda/CUDAContext.h>

        template <typename scalar_t>
        __global__ void inclusive_scan_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            int64_t size,
            int64_t elements_per_group,
            int dim
        ) {
            extern __shared__ scalar_t sdata[];
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int group_id = bid / elements_per_group;
            const int group_size = elements_per_group;

            // Each group processes a block of elements along dimension 'dim'
            int offset = group_id * group_size * size;
            int index = bid % elements_per_group;
            index *= blockDim.x + tid;
            if (index >= size) return;

            // Load data into shared memory
            scalar_t val = input[offset + index];
            sdata[tid] = val;
            __syncthreads();

            // Up-sweep phase (parallel reduction)
            for (int stride = 1; stride <= blockDim.x; stride *= 2) {
                int idx = 2 * stride * tid;
                if (idx < blockDim.x) {
                    sdata[idx + stride] += sdata[idx];
                }
                __syncthreads();
            }

            // Down-sweep phase (exclusive scan)
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                __syncthreads();
                int idx = 2 * stride * tid;
                if (idx < blockDim.x) {
                    scalar_t t = sdata[idx];
                    sdata[idx] += sdata[idx - stride];
                    sdata[idx + stride] = t;
                }
            }

            __syncthreads();

            val = sdata[tid];
            output[offset + index] = val;
        }

        at::Tensor inclusive_scan_cuda(const at::Tensor& input, int64_t dim) {
            const int threads_per_block = 256;
            const int elements_per_group = 1;
            const auto size = input.size(dim);
            const auto blocks_per_group = (size + threads_per_block - 1) / threads_per_block;
            const auto total_blocks = blocks_per_group * elements_per_group;

            auto output = at::empty_like(input);
            AT_DISPATCH_FLOATING_TYPES(input.type(), "inclusive_scan_cuda", ([&] {
                inclusive_scan_kernel<scalar_t>
                <<<total_blocks, threads_per_block, threads_per_block * sizeof(scalar_t), 
                at::cuda::getCurrentCUDAStream()>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    size,
                    elements_per_group,
                    dim
                );
            }));

            return output;
        }

        """
        cpp_sources = """
        at::Tensor inclusive_scan_cuda(const at::Tensor& input, int64_t dim);
        """
        scan_module = load_inline(
            name="scan",
            cpp_sources=cpp_sources,
            cuda_sources=prefix_sum_kernel,
            functions=["inclusive_scan_cuda"],
            verbose=True,
        )
        self.inclusive_scan = scan_module

    def forward(self, x):
        return self.inclusive_scan.inclusive_scan_cuda(x, self.dim)