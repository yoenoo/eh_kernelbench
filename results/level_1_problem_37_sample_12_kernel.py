__global__ void compute_squares_kernel(float* data, float* partial_sums, size_t n) {
    extern __shared__ float block_sums[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Initialize block_sums to 0 first?
    // Or just accumulate

    float sum = 0.0f;
    for (size_t i = bid * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float val = data[i];
        sum += val * val;
    }
    block_sums[tid] = sum;
    __syncthreads();

    // Now perform reduction in the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sums[tid] += block_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = block_sums[0];
    }
}