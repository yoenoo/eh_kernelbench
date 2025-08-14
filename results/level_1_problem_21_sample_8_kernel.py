cuda
__global__ void sigmoid_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float inv_exp = expf(-x[idx]);
        y[idx] = 1.0f / (1.0f + inv_exp);
    }
}