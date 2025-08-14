import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for scaled dot-product attention
scaled_dot_product_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>  // For matrix multiply functions
#include <algorithm>

#define THREADS 256

template <typename scalar_t>
__global__ void scaled_dot_product_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int sequence_length,
    const int embedding_dim,
    const float scale) {

    extern __shared__ char smem[];
    scalar_t* smem_float = reinterpret_cast<scalar_t*>(smem);

    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int q_idx = blockIdx.y;

    const int k_row = head_idx * sequence_length * embedding_dim + q_idx * embedding_dim;
    const int q_row = head_idx * sequence_length * embedding_dim + q_idx * embedding_dim;

    scalar_t* output_row = out + (batch_idx * num_heads + head_idx) * sequence_length * embedding_dim +
                          q_idx * embedding_dim;

    // Compute QK^T
    scalar_t* qk_smem = smem_float;
    scalar_t qk = 0.0;
    #pragma unroll
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
        qk += q[q_row + i] * k[k_row + i];
    }
    __syncwarp();
    qk = blockReduceSum(qk);
    if (threadIdx.x == 0) {
        qk_smem[blockIdx.y] = qk * scale;  // Apply scaling
    }
    __syncthreads();

    // Compute softmax
    scalar_t max_val = qk_smem[blockIdx.y];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (threadIdx.x % warpSize == 0) {
        max_val = __shfl_sync(0xffffffff, max_val, 0);
    }
    scalar_t exp_qk = exp(qk_smem[blockIdx.y] - max_val);
    scalar_t sum = 0.0;
    #pragma unroll
    for (int i = 0; i < sequence_length; i++) {
        sum += exp(qk_smem[i] - max_val);
    }
    __syncthreads();

    // Compute output
    if (threadIdx.x == 0) {
        scalar_t softmax_val = exp_qk / sum;
        for (int k = 0; k < embedding_dim; k++) {
            output_row[k] += softmax_val * v[batch_idx * num_heads * sequence_length * embedding_dim +
                                            head_idx * sequence_length * embedding_dim +
                                            q_idx * embedding_dim + k];
        }
    }
}

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int batch_size,
    int num_heads,
    int sequence_length,
    int embedding_dim,
    float scale) {

    const int total_queries = batch_size * num_heads * sequence_length;
    dim3 blocks(batch_size * num_heads, sequence_length);
    dim3 threads(THREADS);

    auto output = torch::empty({batch_size, num_heads, sequence_length, embedding_dim}, 
                              q.options());

    const size_t shared_mem = sequence_length * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(q.type(), "scaled_dot_product_attention_cuda", ([&]{
        scaled_dot_product_attention_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            q.data<scalar_t>(),
            k.data<scalar_t>(),
            v.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            num_heads,
            sequence_length,
            embedding_dim,
            scale);
    }));

    return output;
}
"""

scaled_dot_product_cpp_source = """
torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int batch_size,
    int num_heads,
    int sequence_length,
    int embedding_dim,
    float scale);
"""

# Compile the custom CUDA kernel
scaled_dot_product = load_inline(
    name="scaled_dot_product",
    cpp_sources=scaled_dot_product_cpp_source,
    cuda_sources=scaled_dot_product_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
    extra_cflags=["-g", "-w"],
    extra_cuda_cflags=["-lineinfo", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scaled_dot_product = scaled_dot_product

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, sequence_length, embedding_dim = Q.size()
        scale = 1.0 / (embedding_dim ** 0.5)
        return self.scaled_dot_product.scaled_dot_product_attention_cuda(
            Q.contiguous(),
            K.contiguous(),
            V.contiguous(),
            batch_size,
            num_heads,
            sequence_length,
            embedding_dim,
            scale
        )