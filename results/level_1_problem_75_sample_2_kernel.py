c++
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void TransposedConv2dKernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int batches, 
    const int in_channels, 
    const int out_channels_per_group,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int out_h, const int out_w) {

    CUDA_1D_KERNEL_LOOP(output_index, batches * out_channels_per_group * out_h * out_w) {
        int w_out = output_index % out_w;
        int h_out = (output_index / out_w) % out_h;
        int c_out = (output_index / (out_w * out_h)) % out_channels_per_group;
        int n = output_index / (out_channels_per_group * out_h * out_w);

        c_out += groups * c_out; // group handling ?

        scalar_t val = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute the effective input position
                const int h_in = (h_out - padding_h - dilation_h * kh) / stride_h;
                const int w_in = (w_out - padding_w - dilation_w * kw) / stride_w;
                
                // Check if the current kernel position is within valid input area
                if (h_in < 0 || h_in >= input.size(2) || w_in < 0 || w_in >= input.size(3)) {
                    continue;
                }

                // Access the input channels for the group
                for (int c_in_group = 0; c_in_group < in_channels / groups; ++c_in_group) {
                    int c_in = c_in_group + (n % groups) * (in_channels / groups);
                    val += weight[n][c_out * kernel_h * kernel_w + kh * kernel_w + kw] * 
                           input[n][c_in][h_in][w_in];
                }
            }
        }
        output[n][c_out][h_out][w_out] = val;
    }
}

std::tuple<torch::Tensor> custom_conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);

    const auto out_channels = weight.size(0) * groups; // Assuming weight is pre-arranged
    const auto out_channels_per_group = out_channels / groups;
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    // Compute output spatial dimensions based on transposed conv formula
    const auto out_h = (in_h - 1) * stride_h - 2 * padding_h + 
                      dilation_h * (kernel_h - 1) + 1;
    const auto out_w = (in_w - 1) * stride_w - 2 * padding_w + 
                      dilation_w * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 threads(256);
    dim3 blocks((batch_size * out_channels_per_group * out_h * out_w + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv_transpose2d", ([&]{
        TransposedConv2dKernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size,
            in_channels,
            out_channels_per_group,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups,
            out_h, out_w);
    }));

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv_transpose2d", &custom_conv_transpose2d, "Custom ConvTranspose2d");
}