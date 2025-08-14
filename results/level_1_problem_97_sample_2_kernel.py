import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled_dot_product_attention
scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void scaled_dot_product_attention_kernel(
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ k,
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len,
    int emb_dim,
    float scale_factor) {

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_idx = threadIdx.x;

    scalar_t score = 0.0;
    __shared__ scalar_t shared_q[256]; // Assuming emb_dim <= 256
    __shared__ scalar_t shared_k[256];

    for (int i = 0; i < seq_len; i += blockDim.x) {
        int k_idx = i + threadIdx.x;
        if (k_idx < seq_len) {
            shared_q[threadIdx.x] = q[batch_idx * num_heads * seq_len * emb_dim + head_idx * seq_len * emb_dim + q_idx * emb_dim + k_idx];
            shared_k[threadIdx.x] = k[batch_idx * num_heads * seq_len * emb_dim + head_idx * seq_len * emb_dim + k_idx * emb_dim + q_idx];
        }
        __syncthreads();

        for (int j = 0; j < blockDim.x && j < seq_len; j++) {
            score += shared_q[j] * shared_k[j];
        }
        __syncthreads();
    }

    score *= scale_factor;
    scalar_t softmax_denom = 0.0;
    for (int i = 0; i < seq_len; i++) {
        scalar_t exp_score = __expf(score - max_score);
        softmax_denom += exp_score;
    }
    scalar_t softmax = exp_score / softmax_denom;

    scalar_t output_val = 0.0;
    for (int i = 0; i < seq_len; i++) {
        output_val += softmax * v[batch_idx * num_heads * seq_len * emb_dim + head_idx * seq_len * emb_dim + i * emb_dim + q_idx];
    }

    out[batch_idx * num_heads * seq_len * emb_dim + head_idx * seq_len * emb_dim + q_idx * emb_dim + q_idx] = output_val;
}

at::Tensor scaled_dot_product_attention_cuda(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    int batch_size,
    int num_heads,
    int seq_len,
    int emb_dim,
    float scale_factor) {

    const int block_size = 256;
    dim3 grid(batch_size, num_heads);
    dim3 block(block_size);

    scaled_dot_product_attention_kernel<float><<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        emb_dim,
        scale_factor);

    return out;
}
"""

scaled_dot_product_attention_cpp_source = """
at::Tensor scaled_dot_product_attention_cuda(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    int batch_size,
    int num_heads,
    int seq_len,
    int emb_dim,
    float scale_factor);
"""

# Compile the inline CUDA code for scaled_dot_product_attention
scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cpp_sources=scaled_dot_product_attention_cpp_source,
    cuda_sources=scaled_dot_product_attention_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
    extra_cflags=["-gencode=arch=compute_75,code=sm_75"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scaled_dot_product_attention = scaled_dot_product_attention

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size = Q.size(0)
        num_heads = Q.size(1)
        seq_len = Q.size(2)
        emb_dim = Q.size(3)
        scale_factor = 1.0 / (emb_dim ** 0.5)
        out = torch.empty_like(Q)
        return self.scaled_dot_product_attention.scaled_dot_product_attention_cuda(
            Q.contiguous(),
            K.contiguous(),
            V.contiguous(),
            batch_size,
            num_heads,
            seq_len,
            emb_dim,
            scale_factor
        )