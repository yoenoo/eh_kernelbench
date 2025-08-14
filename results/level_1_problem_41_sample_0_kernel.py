import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

        # Define and load the custom CUDA kernel
        self.max_pool_1d_cuda = self._load_cuda_op()

    def _load_cuda_op(self):
        max_pool_1d_cuda_source = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void max_pool_1d_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int num_features,
            const int in_seq_len,
            const int out_seq_len,
            const int kernel_size,
            const int stride,
            const int padding,
            const int dilation) {{
            for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * num_features * out_seq_len; idx += blockDim.x * gridDim.x) {{
                const int b = idx / (num_features * out_seq_len);
                const int c = (idx / out_seq_len) % num_features;
                const int o_pos = idx % out_seq_len;

                const int in_start = o_pos * stride - padding;
                int max_val = -FLT_MAX;
                int max_idx = 0;

                for (int k = 0; k < kernel_size; ++k) {{
                    const int in_pos = in_start + k * dilation;
                    if (in_pos < 0 || in_pos >= in_seq_len) {{
                        continue;
                    }}
                    const int in_idx = b * num_features * in_seq_len + c * in_seq_len + in_pos;
                    const scalar_t val = input[in_idx];
                    if (val > max_val) {{
                        max_val = val;
                        max_idx = in_pos;
                    }}
                }}
                const int out_idx = b * num_features * out_seq_len + c * out_seq_len + o_pos;
                output[out_idx] = max_val;
            }}
        }}

        torch::Tensor max_pool_1d(
            torch::Tensor input,
            int64_t kernel_size,
            int64_t stride,
            int64_t padding,
            int64_t dilation) {{
            const auto batch_size = input.size(0);
            const auto num_features = input.size(1);
            const auto in_seq_len = input.size(2);
            const auto out_seq_len = (in_seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

            auto output = torch::empty({{batch_size, num_features, out_seq_len}}, input.options());

            const int threads = 256;
            const int blocks = (batch_size * num_features * out_seq_len + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool_1d_cuda", ([&] {{
                max_pool_1d_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    num_features,
                    in_seq_len,
                    out_seq_len,
                    kernel_size,
                    stride,
                    padding,
                    dilation
                );
            }}));

            return output;
        }}

        """

        max_pool_1d_cpp_source = """
        torch::Tensor max_pool_1d(
            torch::Tensor input,
            int64_t kernel_size,
            int64_t stride,
            int64_t padding,
            int64_t dilation);
        """

        return load_inline(
            name="max_pool_1d_cuda",
            cpp_sources=max_pool_1d_cpp_source,
            cuda_sources=max_pool_1d_cuda_source,
            functions=["max_pool_1d"],
            verbose=True,
            with_cuda=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x.cuda()
        return self.max_pool_1d_cuda.max_pool_1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )