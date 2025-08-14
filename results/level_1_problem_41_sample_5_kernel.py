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
        self.output_padding = 0  # Assuming no output padding needed for forward pass

        # Prepare parameters for CUDA kernel
        self.unfold_padding = padding
        self.unfold_dilation = dilation
        self.unfold_kernel_size = kernel_size
        self.stride_value = self.stride

        # Define custom CUDA MaxPool1d kernel
        cuda_src = """
        #include <torch/extension.h>
        #include <ATen/cuda/CUDAContext.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void custom_max_pool1d_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int num_features,
            const int input_length,
            const int kernel_size,
            const int stride,
            const int padding_l,
            const int dilation,
            const int output_padding) {
            
            const int batch_idx = blockIdx.x;
            const int feature_idx = blockIdx.y;
            const int out_elem = threadIdx.x;

            const int output_length = ((input_length + 2 * padding_l - dilation * (kernel_size - 1) - 1) / stride) + 1 + output_padding;

            for (int o = out_elem; o < output_length; o += blockDim.x) {
                int in_start = o * stride - padding_l;
                scalar_t max_val = -FLT_MAX;
                for (int k = 0; k < kernel_size; ++k) {
                    int in_pos = in_start + dilation * k;
                    if (in_pos < 0 || in_pos >= input_length) {
                        continue;
                    }
                    scalar_t val = input[batch_idx * num_features * input_length + feature_idx * input_length + in_pos];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
                output[batch_idx * num_features * output_length + feature_idx * output_length + o] = max_val;
            }
        }

        torch::Tensor custom_max_pool1d_cuda(torch::Tensor input,
                                            int kernel_size,
                                            int stride,
                                            int padding,
                                            int dilation,
                                            bool return_indices) {
            
            const auto batch_size = input.size(0);
            const auto num_features = input.size(1);
            const auto input_length = input.size(2);
            const auto output_length = 
                (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

            auto output_options = torch::TensorOptions()._like(input);
            auto output = torch::empty({batch_size, num_features, output_length}, output_options);

            const int threads = 256;
            const dim3 blocks(batch_size, num_features, 1);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_max_pool1d_cuda", ([&] {
                custom_max_pool1d_kernel<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch_size,
                    num_features,
                    input_length,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    0); // output_padding is zero
            }));

            cudaDeviceSynchronize();
            return output;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &custom_max_pool1d_cuda, "Custom Max Pool 1D forward");
        }
        """

        self.custom_maxpool = load_inline(
            name="custom_maxpool",
            cpp_sources=[],
            cuda_sources=[cuda_src],
            functions="forward",
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_maxpool.forward(
            x.cuda(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices
        )