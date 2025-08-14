import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for scaled dot-product attention
scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void scaled_dot_product_attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int emb_dim,
    const float scale_factor) {

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_idx = threadIdx.x;

    const int k_start = head_idx * seq_len * emb_dim;
    const int q_offset = (batch_idx * num_heads + head_idx) * seq_len * emb_dim + q_idx * emb_dim;
    const int out_offset = (batch_idx * num_heads + head_idx) * seq_len * emb_dim + q_idx * emb_dim;

    __shared__ scalar_t max_score;
    if (threadIdx.x == 0) {
        max_score = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < seq_len; ++i) {
            int ki = k_start + i * emb_dim;
            scalar_t score = 0;
            for (int e = 0; e < emb_dim; ++e) {
                score += q[q_offset + e] * k[ki + e];
            }
            if (score > max_score) {
                max_score = score;
            }
        }
    }
    __syncthreads();

    scalar_t sum_exp = 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < seq_len; ++i) {
            int ki = k_start + i * emb_dim;
            scalar_t score = 0;
            for (int e = 0; e < emb_dim; ++e) {
                score += q[q_offset + e] * k[ki + e];
            }
            sum_exp += expf((score - max_score) * scale_factor);
        }
    }
    __syncthreads();

    for (int i = 0; i < seq_len; ++i) {
        int ki = k_start + i * emb_dim;
        int vi = (batch_idx * num_heads + head_idx) * seq_len * emb_dim + i * emb_dim;

        scalar_t score = 0;
        for (int e = 0; e < emb_dim; ++e) {
            score += q[q_offset + e] * k[ki + e];
        }
        scalar_t attn_weight = expf((score - max_score) * scale_factor) / sum_exp;

        for (int e = 0; e < emb_dim; ++e) {
            out[out_offset + e] += attn_weight * v[vi + e];
        }
    }
}

at::Tensor scaled_dot_product_attention_forward(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    float scale_factor) {

    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len = q.size(2);
    const int emb_dim = q.size(3);

    auto out = at::zeros({batch_size, num_heads, seq_len, emb_dim}, q.options());

    const int block_dim = 1;
    const dim3 grid_dim(batch_size, num_heads);
    const dim3 block_dim(seq_len);

    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "scaled_dot_product_attention_forward", ([&] {
        scaled_dot_product_attention_forward_kernel<scalar_t><<<grid_dim, block_dim>>>(
            q.data<scalar_t>(),
            k.data<scalar_t>(),
            v.data<scalar_t>(),
            out.data<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            emb_dim,
            scale_factor);
    }));

    return out;
}
"""

scaled_dot_product_attention_cpp_source = """
#include <torch/extension.h>

at::Tensor scaled_dot_product_attention_forward(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    float scale_factor);
"""

scale_factor = 1.0 / (1024**0.5)

scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cpp_sources=scaled_dot_product_attention_cpp_source,
    cuda_sources=scaled_dot_product_attention_source,
    functions=["scaled_dot_product_attention_forward"],
    verbose=True,
    extra_cflags=["-DVERSION_GE_1_5"],
    extra_cuda_cflags=["--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        self.scaled_dot_product_attention = scaled_dot_product_attention

    def forward(self, Q, K, V):
        return self.scaled_dot_product_attention.scaled_dot_product_attention_forward(
            Q, K, V, self.scale_factor
        )