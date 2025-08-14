import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void kl_div_forward_kernel(const float* log_prob, const float* target_prob, float* output, int batch_size, int dim_size) {
    const int batch_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    if (thread_idx >= dim_size) return;

    const int index = batch_idx * dim_size + thread_idx;
    const float p = target_prob[index];
    const float log_q = log_prob[index];
    const float q = exp(log_q); // Since log_prob is log(predictions), compute predictions as exp(log_q)

    // Compute (p * (log_p - log_q)) for each element
    output[index] = p * (log_q - log_q); // Correction: log_p is not available here. Need to use target_prob as p and log_q from predictions
    // Wait, actually target_prob is already the target distribution. However, we need log(p) where p is target_prob.
    // But the original KL Divergence formula is sum(p * log(p/q)). Since log(p/q) is log p - log q.
    // Therefore, log_p = log(target_prob), and log_q is the input log_prob (log(predictions)).

    // The error here was not using log(target_prob) for log_p. To compute this correctly, we need log(target_prob).

    // So the correct term is p * (log_p - log_q) where p = target_prob and log_p = log(target_prob)
    // However, in the current setup, we have target_prob as the input, but not its log. Hence, we need to compute log(target_prob) here.

    // Thus, modifying the kernel to compute log_p on the fly
    const float log_p = logf(target_prob[index]);
    output[index] = p * (log_p - log_q);
}

__global__ void kl_div_reduce_kernel(float* output, int batch_size, int dim_size) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Load data into shared memory
    if (tid < dim_size) {
        shared[tid] = output[bid * dim_size + tid];
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = 1; s < dim_size; s *= 2) {
        int index = 2 * s * tid;
        if (index < dim_size) {
            shared[index] += shared[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[bid] = shared[0];
    }
}

torch::Tensor kl_div_forward_cuda(torch::Tensor log_prob, torch::Tensor target_prob) {
    const int batch_size = log_prob.size(0);
    const int dim_size = log_prob.size(1);

    // Output tensor to store intermediate results (each element's contribution)
    auto output = torch::empty({batch_size, dim_size}, device=log_prob.device());

    dim3 grid(batch_size);
    dim3 block(dim_size); // One thread per element in the dimension

    kl_div_forward_kernel<<<grid, block>>>(log_prob.data_ptr<float>(), target_prob.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim_size);

    // Now, reduce along the dimension (sum over all elements per batch)
    auto result = torch::empty({batch_size}, device=log_prob.device());

    dim3 reduce_grid(batch_size);
    dim3 reduce_block(dim_size); // Threads per block same as dimension size for shared memory

    // Allocate shared memory equal to the dimension size
    kl_div_reduce_kernel<<<reduce_grid, reduce_block, dim_size * sizeof(float)>>>(output.data_ptr<float>(), batch_size, dim_size);

    // Compute mean over the batch as per 'batchmean' reduction
    const float batch_mean_factor = 1.0f / (float(dim_size) * float(batch_size));
    auto mean_result = result.mean() * (float(batch_size) / (float(batch_size) - 1.0f)); // Adjustment for 'batchmean'

    return mean_result;
}

"""

kl_div_cpp_source = """
torch::Tensor kl_div_forward_cuda(torch::Tensor log_prob, torch::Tensor target_prob);
"""

kl_div = load_inline(
    name="kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div

    def forward(self, predictions, targets):
        log_predictions = torch.log(predictions)
        return self.kl_div.kl_div_forward_cuda(log_predictions, targets)