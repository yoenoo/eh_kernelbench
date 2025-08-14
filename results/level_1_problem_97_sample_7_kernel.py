import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled dot-product attention
scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void scaled_dot_product_attention_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ out,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int embed_dim
) {
    extern __shared__ unsigned char smem[];
    T* smem_cast = reinterpret_cast<T*>(smem);

    int head_id = blockIdx.y;
    int batch_id = blockIdx.z;

    int k_idx = batch_id * num_heads * seq_len * embed_dim +
                head_id * seq_len * embed_dim +
                threadIdx.y * embed_dim + threadIdx.x;

    #define QID(b, h, s, e) ((b)*num_heads*seq_len*embed_dim + (h)*seq_len*embed_dim + (s)*embed_dim + (e))
    #define KID(b, h, s, e) ((b)*num_heads*seq_len*embed_dim + (h)*seq_len*embed_dim + (s)*embed_dim + (e))
    #define VID(b, h, s, e) ((b)*num_heads*seq_len*embed_dim + (h)*seq_len*embed_dim + (s)*embed_dim + (e))
    #define OUTID(b, h, s, e) ((b)*num_heads*seq_len*embed_dim + (h)*seq_len*embed_dim + (s)*embed_dim + (e))

    // Load K and V into shared memory for reuse
    if (threadIdx.x < embed_dim && threadIdx.y < seq_len) {
        smem_cast[threadIdx.y * embed_dim + threadIdx.x] = k[k_idx];
    }
    __syncthreads();

    // Iterate over query tokens
    for (int q_idx = threadIdx.x; q_idx < seq_len; q_idx += blockDim.x) {
        T query_val = q[QID(batch_id, head_id, q_idx, threadIdx.y)];
        T sum = 0;

        // Compute dot product with all keys
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            T key_val = smem_cast[k_idx * embed_dim + threadIdx.y];
            sum += query_val * key_val;
        }

        // Scale and store attention scores
        T score = __expf(sum * scale);
        // Here, for simplicity, we assume masking and dropout are handled elsewhere or omitted
        // Apply the attention to values and accumulate
        T val_acc = 0;
        for (int v_idx = 0; v_idx < seq_len; ++v_idx) {
            T val = v[VID(batch_id, head_id, v_idx, threadIdx.x)];
            T attn = score * scale; // Simplified, actual implementation may need proper scaling and masking
            val_acc += attn * val;
        }
        out[OUTID(batch_id, head_id, q_idx, threadIdx.x)] = val_acc;
    }
    #undef QID
    #undef KID
    #undef VID
    #undef OUTID
}

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale
) {
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len = q.size(2);
    const int embed_dim = q.size(3);

    auto out = torch::empty({batch_size, num_heads, seq_len, embed_dim}, 
                            q.options());

    dim3 threads(32, 32); // Tune based on embedding dimension
    dim3 blocks(seq_len, num_heads, batch_size);

    int shared_mem = seq_len * embed_dim * sizeof(torch::scalar_type_to_c_type_t<T>());

    scaled_dot_product_attention_kernel<torch::scalar_type_to_c_type_t<T>> << <blocks, threads, shared_mem >> > (
        q.data_ptr<torch::scalar_type_to_c_type_t<T>>(),
        k.data_ptr<torch::scalar_type_to_c_type_t<T>>(),
        v.data_ptr<torch::scalar_type_to_c_type_t<T>>(),
        out.data_ptr<torch::scalar_type_to_c_type_t<T>>(),
        scale,
        batch_size,
        num_heads,
        seq_len,
        embed_dim
    );

    return out;
}
"""

scaled_dot_product_attention_cpp_source = (
    "torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale);"
)

# Compile the inline CUDA code
scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cpp_sources=scaled_dot_product_attention_cpp_source,
    cuda_sources=scaled_dot_product_attention_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["-arch=sm_75"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scaled_dot_product_attention = scaled_dot_product_attention

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / (Q.size(-1) ** 0.5)
        return self.scaled_dot_product_attention.scaled_dot_product_attention_cuda(Q, K, V, scale)