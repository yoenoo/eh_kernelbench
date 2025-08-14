cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

template <int warps_per_block, int warp_size, int BlockM, int BlockN, int BlockK, typename T>
__global__ void matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* C, int M, int N, int K) {
    constexpr int WarpM = BlockM / (warps_per_block * 4);
    constexpr int WarpN = BlockN / (warps_per_block * 4);
    constexpr int WarpK = BlockK;

    using FragA = cutlass::gemm::GemmFragment<T, BlockM, BlockK, 16>;
    using FragB = cutlass::gemm::GemmFragment<T, BlockK, BlockN, 16>;
    using FragC = cutlass::gemm::GemmFragment<T, BlockM, BlockN, 16>;

    __shared__ typename cutlass::gemm::GemmSharedStorage<
        BlockM, BlockN, BlockK, T, T, cutlass::arch::OpClassTensorOp>::Storage shared_storage;

    cutlass::gemm::threadblock::
        DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T>::IteratorA a_iterator(A, {M, K});
    cutlass::gemm::threadblock::
        DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T>::IteratorB b_iterator(B, {N, K});

    cutlass::gemm::threadblock::
        DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T> mma_op;

    typename cutlass::gemm::threadblock::
        DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T>::FragmentC accum;

    accum.clear();

    for (int batch = 0; batch < (K / BlockK); ++batch) {
        typename cutlass::gemm::threadblock::
            DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T>::FragA smem_a;
        typename cutlass::gemm::threadblock::
            DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T>::FragB smem_b;

        a_iterator.load_shared<typename cutlass::gemm::threadblock::
            DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T>::FragA>(
            shared_storage.A_ref(), batch * BlockK);

        b_iterator.load_shared<typename cutlass::gemm::threadblock::
            DefaultMmaTensorOp<warps_per_block * 4, warp_size, warp_size, WarpM, WarpN, WarpK, T, T, T>::FragB>(
            shared_storage.B_ref(), batch * BlockK);

        __syncthreads();

        mma_op(
            accum,
            smem_a,
            smem_b,
            accum);

        __syncthreads();
    }

    cutlass::epilogue::thread::LinearCombination<
        T, 16, T, T>(
            C + blockIdx.x * BlockM * N + threadIdx.x * 16,
            accum,
            T(1),
            T(0));
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    auto C = torch::zeros({M, N}, A.options());

    const int threads_per_block = 256;
    const dim3 blocks(M * N / (BlockM * BlockN));
    const dim3 threads(threads_per_block);

    matmul_kernel<
        4,
        32,
        128,
        128,
        64,
        float>
        <<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A.cuda(), B.t().cuda())