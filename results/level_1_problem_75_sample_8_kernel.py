#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void transposed_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_h,
    const int kernel_w,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int bias_enable,
    const scalar_t* __restrict__ bias) {

  CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels * output_height * output_width) {
    int w_out = output_idx % output_width;
    int h_out = (output_idx / output_width) % output_height;
    int c_out_group = (output_idx / (output_width * output_height)) % (out_channels / groups);
    int group_idx = output_idx / (output_width * output_height * (out_channels / groups));
    int n = output_idx / (output_height * output_width * out_channels);

    int c_in = group_idx * (in_channels / groups);
    const int c_out_offset = group_idx * (out_channels / groups);

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        // Compute input coordinates
        int h_in = (h_out * stride_h - padding_h - kh * dilation_h);
        int w_in = (w_out * stride_w - padding_w - kw * kernel_w); // kernel_w? 

        if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
          continue;
        }

        for (int c_in_offset = 0; c_in_offset < (in_channels / groups); ++c_in_offset) {
          int k_idx = (kh * kernel_w + kw) * in_channels + (c_in + c_in_offset);
          int in_offset = (n * in_channels + c_in + c_in_offset) * input_height * input_width
                          + h_in * input_width + w_in;
          val += input[in_offset] * weight[k_idx + c_out_group * kernel_h * kernel_w * in_channels];
        }
      }
    }

    output[output_idx] = val;
    if (bias_enable) {
      output[output_idx] += bias[c_out_group + c_out_offset];
    }
  }
}

std::vector<int64_t> compute_output_size(
    const int64_t input_height, const int64_t input_width,
    const int64_t kernel_h, const int64_t kernel_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t padding_h, const int64_t padding_w,
    const int64_t dilation_h, const int64_t dilation_w) {

  const int64_t output_h = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
  const int64_t output_w = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;
  return {output_h, output_w};
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    bool bias_enable) {

  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  const int out_channels = weight.size(0) / groups;
  const int kernel_h = weight.size(2);
  const int kernel_w = weight.size(3);

  const int input_height = input.size(2);
  const int input_width = input.size(3);

  auto output_dims = compute_output_size(
      input_height, input_width,
      kernel_h, kernel_w,
      stride_h, stride_w,
      padding_h, padding_w,
      dilation_h, dilation_w);

  const int output_h = output_dims[0];
  const int output_w = output_dims[1];

  auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

  const int num_output_elements = batch_size * out_channels * output_h * output_w;

  dim3 blocks((num_output_elements + 1024 - 1) / 1024);
  dim3 threads(1024);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "transposed_conv2d_cuda", ([&] {
    transposed_conv2d_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        output_h,
        output_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        (bias_enable) ? 1 : 0,
        (bias_enable) ? bias.data_ptr<scalar_t>() : nullptr);
  }));

  cudaDeviceSynchronize();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transposed_conv2d_cuda", &transposed_conv2d_cuda, "Transposed Convolution CUDA");
}