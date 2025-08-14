cuda
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                             int M, int K, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}