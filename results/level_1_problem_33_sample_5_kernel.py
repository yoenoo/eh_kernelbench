__global__ void compute_mean(const float* __restrict__ input,
                            float* mean_out,
                            int batch_size,
                            int channels,
                            int dim1,
                            int dim2) {
    int c = blockIdx.x;
    if (c >= channels) return; // in case of more blocks than channels

    const int N = batch_size * dim1 * dim2; // total elements per channel
    extern __shared__ float partial_sums[];

    int tid = threadIdx.x;
    partial_sums[tid] = 0.0f;

    // each thread computes its own partial sum
    for (int i = tid; i < N; i += blockDim.x) {
        // compute batch, h, w
        int b = i / (dim1 * dim2);
        int rem = i % (dim1 * dim2);
        int h = rem / dim2;
        int w = rem % dim2;
        // compute offset
        int offset = b * channels * dim1 * dim2 +
                     c * dim1 * dim2 +
                     h * dim2 + w;
        partial_sums[tid] += input[offset];
    }

    __syncthreads();

    // now perform reduction in shared memory
    for (int s = blockDim.x/2; s >0; s>>=1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean_out[c] = partial_sums[0] / static_cast<float>(N);
    }
}