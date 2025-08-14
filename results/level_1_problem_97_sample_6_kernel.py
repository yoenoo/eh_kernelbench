import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void scaled_dot_product_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int batch_size,
    int num_heads,
    int sequence_length,
    int embedding_dimension) {

    int batch = blockIdx.x;
    int head = blockIdx.y;
    int q_idx = threadIdx.x;

    const int k_idx = threadIdx.y;
    __shared__ scalar_t cache[32][32]; // Assuming max thread block size

    scalar_t sum = 0;
    for (int i = 0; i < sequence_length; i += blockDim.y) {
        int k_offset = i + k_idx;
        if (k_offset < sequence_length) {
            scalar_t dot = 0;
            for (int e = 0; e < embedding_dimension; e++) {
                dot += q[batch * num_heads * sequence_length * embedding_dimension +
                        head * sequence_length * embedding_dimension +
                        q_idx * embedding_dimension + e] *
                       k[batch * num_heads * sequence_length * embedding_dimension +
                         head * sequence_length * embedding_dimension +
                         k_offset * embedding_dimension + e];
            }
            cache[threadIdx.x][k_idx] = dot;
            __syncthreads();

            // Row-wise reduction for softmax denominator
            if (threadIdx.x == q_idx && k_idx == 0) {
                scalar_t max_val = -FLT_MAX;
                for (int j = 0; j < sequence_length; j++) {
                    if (cache[threadIdx.x][j] > max_val) {
                        max_val = cache[threadIdx.x][j];
                    }
                }
                sum = 0;
                for (int j = 0; j < sequence_length; j++) {
                    scalar_t exp_val = exp(cache[threadIdx.x][j] - max_val);
                    sum += exp_val;
                    cache[threadIdx.x][j] = exp_val;
                }
            }
            __syncthreads();

            if (k_offset < sequence_length) {
                scalar_t weight = cache[threadIdx.x][k_idx] / sum;
                for (int e = 0; e < embedding_dimension; e++) {
                    out[batch * num_heads * sequence_length * embedding_dimension +
                        head * sequence_length * embedding_dimension +
                        q_idx * embedding_dimension + e] +=
                        weight * v[batch * num_heads * sequence_length * embedding_dimension +
                                   head * sequence_length * embedding_dimension +
                                   k_offset * embedding_dimension + e];
                }
            }
            __syncthreads();
        }
    }
}

torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int sequence_length = q.size(2);
    const int embedding_dimension = q.size(3);

    auto out = torch::empty({batch_size, num_heads, sequence_length, embedding_dimension}, 
                            q.options());

    dim3 threads(32, 32); // Adjust thread block size based on SM capacity
    dim3 blocks(batch_size, num_heads);

    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "scaled_dot_product_attention_cuda", ([&] {
        scaled_dot_product_attention_kernel<scalar_t><<<blocks, threads>>>(
            q.data<scalar_t>(),
            k.data<scalar_t>(),
            v.data<scalar_t>(),
            out.data<scalar_t>(),
            batch_size,
            num_heads,
            sequence_length,
            embedding_dimension);
    }));

    cudaDeviceSynchronize();
    return out;
}
"""

        scaled_dot_product_attention_cpp_source = """
        torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
        """

        self.scaled_dot_product_attention = load_inline(
            name="scaled_dot_product_attention",
            cpp_sources=scaled_dot_product_attention_cpp_source,
            cuda_sources=scaled_dot_product_attention_source,
            functions=["scaled_dot_product_attention_cuda"],
            verbose=True
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.scaled_dot_product_attention.scaled_dot_product_attention_cuda(Q, K, V)