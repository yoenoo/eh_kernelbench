#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void fused_conv3d_relu_kernel(const scalar_t* __restrict__ input,
                                   const scalar_t* __restrict__ weight,
                                   scalar_t* __restrict__ output,
                                   const int output_size,
                                   const int in_channels,
                                   const int out_channels,
                                   const int kernel_w, const int kernel_h, const int kernel_d,
                                   const int input_h, const int input_w, const int input_d,
                                   const int output_h, const int output_w, const int output_d) {

    CUDA_KERNEL_LOOP(index, output_size) {
        int d_out = index % output_d;
        int h_out = (index / output_d) % output_h;
        int w_out = (index / (output_d * output_h)) % output_w;
        int c_out = index / (output_d * output_h * output_w);

        scalar_t sum = 0;
        for (int k_d = 0; k_d < kernel_d; ++k_d) {
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int in_offset = c_in * input_w * input_h * input_d
                                      + (k_w + w_out) * input_h * input_d
                                      + (k_h + h_out) * input_d
                                      + (k_d + d_out);
                        int weight_offset = c_out * in_channels * kernel_w * kernel_h * kernel_d
                                          + c_in * kernel_w * kernel_h * kernel_d
                                          + k_w * kernel_h * kernel_d
                                          + k_h * kernel_d
                                          + k_d;
                        sum += input[in_offset] * weight[weight_offset];
                    }
                }
            }
        }
        output[index] = fmaxf(sum, 0); // ReLU activation
    }
}

std::tuple<torch::Tensor, torch::Tensor> fused_conv3d_relu(
    torch::Tensor input,
    torch::Tensor weight,
    const int kernel_w,
    const int kernel_h,
    const int kernel_d) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_w = input.size(2);
    const int input_h = input.size(3);
    const int input_d = input.size(4);

    const int out_channels = weight.size(0);
    const int output_w = input_w - kernel_w + 1;
    const int output_h = input_h - kernel_h + 1;
    const int output_d = input_d - kernel_d + 1;

    int output_size = batch_size * out_channels * output_w * output_h * output_d;
    torch::Tensor output = torch::empty({batch_size, out_channels, output_w, output_h, output_d}, input.options());

    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv3d_relu", ([&] {
        fused_conv3d_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            output_size,
            in_channels,
            out_channels,
            kernel_w,
            kernel_h,
            kernel_d,
            input_h,
            input_w,
            input_d,
            output_h,
            output_w,
            output_d);
    }));

    return std::make_tuple(output, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv3d_relu", &fused_conv3d_relu, "Fused 3D convolution and ReLU");
}