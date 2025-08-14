#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

using torch::Tensor;

#define BLOCK_SIZE 32

template <typename scalar_t>
__global__ void DepthwiseConv2DForward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int height,
    int width,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w) {

    const int b = blockIdx.z;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_y = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= in_channels || out_y >= out_height) return;

    int output_index = c + in_channels * (out_y + out_height * b);
    output[output_index] = 0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        int h = out_y * stride_h - padding_h + kh * dilation_h;
        if (h < 0 || h >= height) continue;

        for (int kw = 0; kw < kernel_w; ++kw) {
            int w = threadIdx.x * stride_w - padding_w + kw * dilation_w;
            if (w < 0 || w >= width) continue;

            int input_offset = c + in_channels * (h + height * (w + width * b));
            int weight_offset = kw + kernel_w * (kh + kernel_h * c);
            output[output_index] += input[input_offset] * weight[weight_offset];
        }
    }

    // Apply ReLU activation
    output[output_index] = fmax(output[output_index], 0);
}

class DepthwiseConvReLUFunction(torch::autograd::Function) {
public:
    static torch::Tensor forward(
        torch::autograd::AutoGradAllocator* allocator,
        torch::Tensor input,
        torch::Tensor weight,
        int stride_h,
        int stride_w,
        int padding_h,
        int padding_w,
        int dilation_h,
        int dilation_w) {
        
        const auto batch_size = input.size(0);
        const auto in_channels = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);
        const auto kernel_h = weight.size(2);
        const auto kernel_w = weight.size(3);

        auto output_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        auto output_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        auto output = at::empty({batch_size, in_channels, output_height, output_width}, input.options());

        dim3 threads(BLOCK_SIZE, 1, 1);
        dim3 blocks(output_width, in_channels, batch_size);

        AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv_relu", ([&] {
            DepthwiseConv2DForward<scalar_t><<<blocks, threads>>>(
                input.data<scalar_t>(),
                weight.data<scalar_t>(),
                output.data<scalar_t>(),
                batch_size,
                in_channels,
                output_height,
                output_width,
                kernel_h,
                kernel_w,
                height,
                width,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w);
        }));

        return output;
    }

    static torch::autograd::Variable backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variables::const_variables_t grad_output) {
        // Implement backward kernel here (simplified for brevity)
        // Returning dummy gradients for demonstration purposes
        auto grad_input = at::zeros_like(ctx->inputs()[0]);
        auto grad_weight = at::zeros_like(ctx->saved_variables()[0]);
        return {grad_input, grad_weight};
    }
};

torch::Tensor DepthwiseConvReLUForward(torch::Tensor input,
                                      torch::Tensor weight,
                                      int stride_h,
                                      int stride_w,
                                      int padding_h,
                                      int padding_w,
                                      int dilation_h,
                                      int dilation_w) {
    return DepthwiseConvReLUFunction::apply(input, weight, 
                                            stride_h, stride_w, 
                                            padding_h, padding_w, 
                                            dilation_h, dilation_w);
}