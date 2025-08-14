import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled dot-product attention
scale_attention_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void scaled_dot_product_attention_kernel(
    const float* q, const float* k, const float* v,
    float* out,
    int batch_size, int num_heads, int seq_len, int emb_dim,
    float scale_factor) {

    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int q_idx = threadIdx.x;

    if (q_idx >= seq_len) return;

    float sum = 0.0;
    float max_val = -INFINITY;

    // Compute the scaled dot product for all k
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        float dot_product = 0.0;
        for (int d = 0; d < emb_dim; ++d) {
            dot_product += q[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + q_idx * emb_dim + d] *
                        k[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + k_idx * emb_dim + d];
        }
        float scaled = dot_product * scale_factor;
        if (scaled > max_val) max_val = scaled;
    }

    // Compute exp and sum for softmax denominator
    float denominator_sum = 0.0;
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        float dot_product = 0.0;
        for (int d = 0; d < emb_dim; ++d) {
            dot_product += q[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + q_idx * emb_dim + d] *
                        k[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + k_idx * emb_dim + d];
        }
        float scaled = dot_product * scale_factor;
        float exp_val = expf(scaled - max_val);
        denominator_sum += exp_val;
    }

    // Compute the weighted sum
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        float dot_product = 0.0;
        for (int d = 0; d < emb_dim; ++d) {
            dot_product += q[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + q_idx * emb_dim + d] *
                        k[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + k_idx * emb_dim + d];
        }
        float scaled = dot_product * scale_factor;
        float weight = expf(scaled - max_val) / denominator_sum;
        for (int d = 0; d < emb_dim; ++d) {
            out[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + q_idx * emb_dim + d] +=
                weight * v[batch_id * num_heads * seq_len * emb_dim + head_id * seq_len * emb_dim + k_idx * emb_dim + d];
        }
    }
}

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    int batch_size, int num_heads, int seq_len, int emb_dim) {

    const int threads_per_block = seq_len;
    const dim3 blocks(batch_size, num_heads);
    const dim3 threads(threads_per_block);

    float scale_factor = 1.0 / sqrt(emb_dim);

    torch::Tensor out = torch::zeros({batch_size, num_heads, seq_len, emb_dim}, 
                                    torch::device("cuda").dtype(torch::kFloat16));

    scaled_dot_product_attention_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, num_heads, seq_len, emb_dim, scale_factor);

    return out;
}
"""

scale_attention_cpp_source = (
    "torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, int batch_size, int num_heads, int seq_len, int emb_dim);"
)

# Compile the inline CUDA code for scaled dot-product attention
scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cpp_sources=scale_attention_cpp_source,
    cuda_sources=scale_attention_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
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
        return self.scaled_dot_product_attention.scaled_dot_product_attention_cuda(Q, K, V, batch_size, num_heads, seq_len, emb_dim)