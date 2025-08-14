import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused scaled dot-product attention
scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void scaled_dot_product_attention_forward(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int emb_dim,
    const float scale) {

    int batch = blockIdx.x;
    int head = blockIdx.y;
    int q_idx = threadIdx.x;

    __shared__ scalar_t cache[512][512]; // Assuming seq_len <= 512

    if (q_idx < seq_len) {
        scalar_t sum = 0.0;
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            scalar_t dot = 0.0;
            for (int d = 0; d < emb_dim; ++d) {
                dot += q[batch * num_heads * seq_len * emb_dim + head * seq_len * emb_dim + q_idx * emb_dim + d] *
                       k[batch * num_heads * seq_len * emb_dim + head * seq_len * emb_dim + k_idx * emb_dim + d];
            }
            cache[q_idx][k_idx] = __expf(dot * scale);
            sum += cache[q_idx][k_idx];
        }
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            cache[q_idx][k_idx] /= sum;
        }
    }
    __syncthreads();

    if (q_idx < seq_len) {
        scalar_t output_val = 0.0;
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            output_val += cache[q_idx][k_idx] *
                          v[batch * num_heads * seq_len * emb_dim + head * seq_len * emb_dim + k_idx * emb_dim + q_idx];
        }
        out[batch * num_heads * seq_len * emb_dim + head * seq_len * emb_dim + q_idx * emb_dim + q_idx] = output_val;
    }
}

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len = q.size(2);
    const int emb_dim = q.size(3);
    const float scale = 1.0 / sqrt(emb_dim);

    auto out = torch::empty_like(q);

    dim3 blocks(batch_size, num_heads);
    dim3 threads(seq_len);

    scaled_dot_product_attention_forward<float><<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        emb_dim,
        scale
    );

    return out;
}
"""

scaled_dot_product_attention_cpp_source = (
    "torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);"
)

# Compile the inline CUDA code
attention_op = load_inline(
    name="scaled_dot_product_attention",
    cpp_sources=scaled_dot_product_attention_cpp_source,
    cuda_sources=scaled_dot_product_attention_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.attention_op = attention_op

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.attention_op.scaled_dot_product_attention_cuda(Q, K, V)