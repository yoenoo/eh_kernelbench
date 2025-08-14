import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

template <typename T>
__global__ void scaled_dot_product_attention_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len,
    int embed_dim,
    float scale) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.z;

    const int k_batch_offset = batch_idx * num_heads * seq_len * embed_dim;
    const int q_offset = k_batch_offset + head_idx * seq_len * embed_dim + q_idx * embed_dim;
    const int v_offset = k_batch_offset + head_idx * seq_len * embed_dim;

    __shared__ T max_val;
    __shared__ T sum;

    T temp = -FLT_MAX;
    for (int k_idx = threadIdx.x; k_idx < seq_len; k_idx += blockDim.x) {
        T dot = 0;
        for (int e = 0; e < embed_dim; ++e) {
            dot += q[q_offset + e] * k[k_batch_offset + head_idx * seq_len * embed_dim + k_idx * embed_dim + e];
        }
        temp = max(temp, dot);
    }

    max_val = blockReduceMax(temp);
    __syncthreads();

    T exp_sum = 0;
    for (int k_idx = threadIdx.x; k_idx < seq_len; k_idx += blockDim.x) {
        T dot = 0;
        for (int e = 0; e < embed_dim; ++e) {
            dot += q[q_offset + e] * k[k_batch_offset + head_idx * seq_len * embed_dim + k_idx * embed_dim + e];
        }
        T weighted_v = expf((dot - max_val) * scale);
        atomicAdd(&exp_sum, weighted_v);
        for (int e = 0; e < embed_dim; ++e) {
            atomicAdd(&out[batch_idx * num_heads * seq_len * embed_dim + head_idx * seq_len * embed_dim + q_idx * embed_dim + e],
                weighted_v * v[v_offset + k_idx * embed_dim + e]);
        }
    }
    __syncthreads();
}

at::Tensor scaled_dot_product_attention_cuda(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v) {
    
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len = q.size(2);
    const int embed_dim = q.size(3);

    auto out = at::empty({batch_size, num_heads, seq_len, embed_dim}, q.options());

    const int block_size = 256;
    dim3 blocks(batch_size, num_heads, seq_len);
    dim3 threads(block_size);

    scaled_dot_product_attention_kernel<float><<<blocks, threads>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, num_heads, seq_len, embed_dim,
        1.0f / sqrt(embed_dim));

    return out;
}

"""

elementwise_add_cpp_source = (
    "at::Tensor scaled_dot_product_attention_cuda(at::Tensor q, at::Tensor k, at::Tensor v);"
)

scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cuda_sources=elementwise_add_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scaled_dot_product_attention_cuda = scaled_dot_product_attention

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.scaled_dot_product_attention_cuda.scaled_dot_product_attention_cuda(Q, K, V)