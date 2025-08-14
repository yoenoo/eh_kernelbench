__global__ void leaky_relu_kernel(const float* input, float* output, float slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val > 0) ? val : val * slope;
    }
}