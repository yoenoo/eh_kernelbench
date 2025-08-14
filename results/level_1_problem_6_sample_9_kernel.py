import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mmul_kernel = load_inline(
            name="mmul_kernel",
            cuda_sources="""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <mma.h>

                template <typename T>
                __global__ void mmul_kernel(
                    const T* __restrict__ a,
                    const T* __restrict__ b,
                    T* __restrict__ c,
                    int m,
                    int n,
                    int k
                ) {
                    using mma_op = typename std::conditional<
                        std::is_same<T, float>::value,
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major>,
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major>
                    >::type;

                    extern __shared__ char s_a[];
                    auto s_a_mat = reinterpret_cast<T*>(s_a);
                    auto s_b_mat = s_a_mat + (16 * 16);

                    nvcuda::wmma::float16* s_c;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_c, 16, 16, 16, T> c_frag;

                    int tx = threadIdx.x;
                    int ty = threadIdx.y;
                    int bx = blockIdx.x;
                    int by = blockIdx.y;

                    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

                    for (int batch = 0; batch < (k + 15)/16; ++batch) {
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T> a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T> b_frag;

                        int a_row = by * 16 + ty;
                        int a_col = batch * 16 + tx;
                        int b_row = batch * 16 + ty;
                        int b_col = bx * 16 + tx;

                        if (a_row < m && a_col < k) {
                            s_a_mat[ty * 16 + tx] = a[a_row * k + a_col];
                        }
                        if (b_row < k && b_col < n) {
                            s_b_mat[ty * 16 + tx] = b[b_row * n + b_col];
                        }

                        __syncthreads();

                        nvcuda::wmma::load_matrix_sync(a_frag, s_a_mat, 16);
                        nvcuda::wmma::load_matrix_sync(b_frag, s_b_mat, 16);

                        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

                        __syncthreads();
                    }

                    int c_row = by * 16 + ty;
                    int c_col = bx * 16 + tx;
                    if (c_row < m && c_col < n) {
                        c[c_row * n + c_col] = c_frag[ty * 16 + tx];
                    }
                }

                at::Tensor mmul_cuda(
                    at::Tensor a,
                    at::Tensor b,
                    int m,
                    int n,
                    int k
                ) {
                    auto options = a.options();
                    auto c = at::zeros({m, n}, options);

                    dim3 threads(16, 16);
                    dim3 blocks((n + 15)/16, (m + 15)/16);

                    int sm_size = 2 * 16 * 16 * sizeof(float) + 16 * 16 * sizeof(nvcuda::wmma::float16);
                    mmul_kernel<float><<<blocks, threads, sm_size>>>(
                        a.data_ptr<float>(),
                        b.data_ptr<float>(),
                        c.data_ptr<float>(),
                        m, n, k
                    );

                    return c;
                }
            """,
            functions=["mmul_cuda"],
            verbose=True
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.mmul_kernel.mmul_cuda(A, B, A.size(0), B.size(1), A.size(1))