import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class DepthwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, stride, padding, dilation):
        batch_size, in_channels, height, width = x.shape
        kernel_size = weight.size(2)
        out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        out_width = width

        output = torch.empty(batch_size, in_channels, out_height, out_width, device=x.device)
        num_threads = 512
        blocks = ((out_height * out_width * in_channels + num_threads - 1) // num_threads)
        
        depthwise_conv2d_kernel = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        __global__ void depthwise_conv2d_forward(const float* x, const float* weight, float* output,
                                                 int batch_size, int in_channels, int height, int width,
                                                 int kernel_size, int out_height, int out_width,
                                                 int stride, int padding, int dilation) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * in_channels * out_height * out_width) return;

            int w = idx % out_width;
            int h = (idx / out_width) % out_height;
            int c = (idx / out_width / out_height) % in_channels;
            int n = idx / (in_channels * out_height * out_width);

            int pad_h = padding;
            int pad_w = 0; // padding only on height side as kernel is asymmetric (kernel_size x 1)

            int output_idx = n * in_channels * out_height * out_width + c * out_height * out_width + h * out_width + w;

            float acc = 0.0;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int in_h = h * stride - pad_h + kh * dilation;
                if (in_h < 0 || in_h >= height) continue;
                int in_w = w;
                int x_idx = n * in_channels * height * width + c * height * width + in_h * width + in_w;
                int weight_idx = c * kernel_size + kh;
                acc += x[x_idx] * weight[weight_idx];
            }
            output[output_idx] = acc;
        }
        """

        depthwise_conv2d = load_inline(
            name="depthwise_conv2d",
            cpp_sources="",
            cuda_sources=depthwise_conv2d_kernel,
            functions=[],
            verbose=False,
            with_cuda=True
        )

        depthwise_conv2d_kernel_func = depthwise_conv2d.get_function("depthwise_conv2d_forward")

        depthwise_conv2d_kernel_func(block=(num_threads,1,1), grid=(blocks,1,1), streams=[torch.cuda.current_stream().cuda_stream], 
            args=[
                x.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                batch_size, in_channels, height, width,
                kernel_size, out_height, out_width,
                stride, padding, dilation
            ])
        
        ctx.save_for_backward(x, weight)
        ctx.params = (stride, padding, dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.params
        batch_size, in_channels, out_height, out_width = grad_output.shape
        kernel_size = weight.size(2)
        height = (out_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)

        num_threads = 512
        blocks = ((x.numel() + num_threads - 1) // num_threads)

        depthwise_conv2d_bwd_kernel = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        __global__ void depthwise_conv2d_backward_weights(const float* x, const float* grad_output, float* grad_weight,
                                                          int batch_size, int in_channels, int height, int width,
                                                          int kernel_size, int out_height, int out_width,
                                                          int stride, int padding, int dilation) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= in_channels * kernel_size) return;

            int c = idx / kernel_size;
            int kh = idx % kernel_size;

            float grad = 0.0;
            for (int n = 0; n < batch_size; ++n) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        int in_h = h * stride - padding + kh * dilation;
                        if (in_h < 0 || in_h >= height) continue;
                        int x_idx = n * in_channels * height * width + c * height * width + in_h * width + w;
                        int go_idx = n * in_channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
                        grad += x[x_idx] * grad_output[go_idx];
                    }
                }
            }
            grad_weight[idx] = grad;
        }

        __global__ void depthwise_conv2d_backward_input(const float* grad_output, const float* weight, float* grad_x,
                                                        int batch_size, int in_channels, int height, int width,
                                                        int kernel_size, int out_height, int out_width,
                                                        int stride, int padding, int dilation) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * in_channels * height * width) return;

            int w = idx % width;
            int h = (idx / width) % height;
            int c = (idx / width / height) % in_channels;
            int n = idx / (in_channels * height * width);

            float acc = 0.0;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int out_h = (h - padding + kh * dilation) / stride;
                if (out_h < 0 || out_h >= out_height || (h - padding + kh * dilation) % stride != 0) continue;
                int out_w = w;
                int go_idx = n * in_channels * out_height * out_width + c * out_height * out_width + out_h * out_width + out_w;
                int kW_idx = c * kernel_size + kh;
                acc += grad_output[go_idx] * weight[kW_idx];
            }
            grad_x[idx] = acc;
        }
        """

        depthwise_conv2d_bwd = load_inline(
            name="depthwise_conv2d_bwd",
            cpp_sources="",
            cuda_sources=depthwise_conv2d_bwd_kernel,
            functions=[],
            verbose=False,
            with_cuda=True
        )

        # Compute gradient w.r.t weights
        depthwise_conv2d_bwd.get_function("depthwise_conv2d_backward_weights")(
            block=(num_threads,1,1), grid=(blocks,1,1), streams=[torch.cuda.current_stream().cuda_stream], 
            args=[
                x.data_ptr(),
                grad_output.data_ptr(),
                grad_weight.data_ptr(),
                batch_size, in_channels, height, width,
                kernel_size, out_height, out_width,
                stride, padding, dilation
            ])

        # Compute gradient w.r.t input
        depthwise_conv2d_bwd.get_function("depthwise_conv2d_backward_input")(
            block=(num_threads,1,1), grid=(blocks,1,1), streams=[torch.cuda.current_stream().cuda_stream],
            args=[
                grad_output.data_ptr(),
                weight.data_ptr(),
                grad_x.data_ptr(),
                batch_size, in_channels, height, width,
                kernel_size, out_height, out_width,
                stride, padding, dilation
            ])

        return grad_x, grad_weight, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Flatten spatial dimensions since kernel is 1 in width dimension
        weight = self.weight.squeeze(-1).squeeze(-1) # Convert to 1D kernel [C, K]
        output = DepthwiseConv2dFunction.apply(x, weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output