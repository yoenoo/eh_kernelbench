#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void custom_conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels_per_group,
    const int height_in,
    const int width_in,
    const int height_kernel,
    const int width_kernel,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int height_out,
    const int width_out) {

    const int h_out = blockIdx.y;
    const int w_out = blockIdx.x;
    const int group = blockIdx.z;

    const int out_channels_start = group * out_channels_per_group;

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size) {
        scalar_t sum = 0;
        const int in_channel_start = group * (in_channels / groups);

        for (int kh = 0; kh < height_kernel; ++kh) {
            for (int kw = 0; kw < width_kernel; ++kw) {
                // Compute input spatial coordinates
                const int h_in = (h_out - kh * dilation_h + 2 * padding_h) / stride_h;
                const int w_in = (w_out - kw * dilation_w + 2 * padding_w) / stride_w;

                // Check if input coordinates are valid
                if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                    for (int ic = 0; ic < (in_channels / groups); ++ic) {
                        const int weight_index = (out_channels_start + kh * width_kernel + kw) * (in_channels / groups) + ic;
                        const int input_index = output_idx * in_channels * height_in * width_in +
                            (in_channel_start + ic) * height_in * width_in +
                            h_in * width_in + w_in;
                        sum += weight[weight_index] * input[input_index];
                    }
                }
            }
        }

        const int out_offset = output_idx * out_channels * height_out * width_out +
            out_channels_start * height_out * width_out +
            h_out * width_out + w_out;
        atomicAdd(&output[out_offset], sum);
    }
}

std::tuple<torch::Tensor> custom_conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);

    const int out_channels = weight.size(0);
    const int height_kernel = weight.size(2);
    const int width_kernel = weight.size(3);

    const int out_channels_per_group = out_channels / groups;

    // Compute output dimensions
    const int height_out = (height_in - 1) * stride_h - 2 * padding_h + 
        dilation_h * (height_kernel - 1) + 1;
    const int width_out = (width_in - 1) * stride_w - 2 * padding_w +
        dilation_w * (width_kernel - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, 
                              input.options());

    const dim3 blocks(width_out, height_out, groups);
    const dim3 threads(128); // Tuned based on SM count

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv_transpose2d", ([&] {
        custom_conv_transpose2d_kernel<scalar_t><<<blocks, threads, 0, 
                at::cuda::getCurrentCUDAStream()>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels_per_group,
            height_in,
            width_in,
            height_kernel,
            width_kernel,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            height_out,
            width_out);
    }));

    return std::make_tuple(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv_transpose2d", &custom_conv_transpose2d, "Custom ConvTranspose2D");
}