cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void log_softmax_kernel(const float* input, float* output, int batch_size, int dim_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_stride = gridDim.x * blockDim.x;

    if (batch_idx < batch_size) {
        const auto* input_row = input + batch_idx * dim_size;
        float* output_row = output + batch_idx * dim_size;

        // Compute the max for this row to prevent overflow
        float max_val = -FLT_MAX;
        for (int i = 0; i < dim_size; ++i) {
            max_val = fmaxf(max_val, input_row[i]);
        }

        // Compute the numerator (exp(x - max)) and the denominator sum
        float sum = 0.0f;
        for (int i = 0; i < dim_size; ++i) {
            sum += expf(input_row[i] - max_val);
        }

        // Compute log_softmax = (x_i - max) - log(sum)
        for (int i = 0; i < dim_size; ++i) {
            output_row[i] = (input_row[i] - max_val) - logf(sum);
        }
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1);

    const int threads_per_block = 256;
    const int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    auto output = torch::empty_like(input);
    log_softmax_kernel<<<num_blocks, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim_size);

    return output;
}