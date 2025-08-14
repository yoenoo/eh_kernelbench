import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class GroupNormCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, num_groups, eps):
        output = torch.empty_like(input)
        mean = torch.empty_like(input)
        var = torch.empty_like(input)
        
        # Dimensions and parameters
        n, c, h, w = input.shape
        group_size = c // num_groups
        
        threads = 256
        blocks = (c * h * w + threads - 1) // threads
        
        stream = torch.cuda.current_stream().stream
        load_groupnorm_forward(stream, input, weight, bias, output, mean, var, 
                              num_groups, group_size, n, h, w, eps)
        
        ctx.save_for_backward(input, weight, bias, mean, var)
        ctx.num_groups = num_groups
        ctx.eps = eps
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mean, var = ctx.saved_tensors
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        
        num_groups = ctx.num_groups
        eps = ctx.eps
        n, c, h, w = input.shape
        group_size = c // num_groups
        
        threads = 256
        blocks = (c * h * w + threads - 1) // threads
        
        stream = torch.cuda.current_stream().stream
        load_groupnorm_backward(stream, grad_output, input, weight, bias, mean, var, 
                               grad_input, grad_weight, grad_bias, num_groups, group_size, n, h, w, eps)
        
        return grad_input, grad_weight, grad_bias, None, None
    
def load_groupnorm_forward(stream, input, weight, bias, output, mean, var, 
                          num_groups, group_size, n, h, w, eps):
    # CUDA kernel code for forward pass
    kernel_code = f"""
    extern "C" __global__ void groupnorm_forward(float* input, float* weight, float* bias, 
                                                 float* output, float* mean, float* var,
                                                 int num_groups, int group_size, int n, int h, int w, float eps) {{
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= n*h*w*num_groups*group_size) return;
        
        // Calculate group index and channel within group
        int group = c / group_size;
        int channel_in_group = c % group_size;
        
        // Compute mean and variance for each group
        float sum = 0.0;
        for(int i=0; i<h*w; i++) {{
            sum += input[channel_in_group + i*group_size + group*group_size*h*w];
        }}
        float mean_val = sum / (h*w);
        float var_sum = 0.0;
        for(int i=0; i<h*w; i++) {{
            float x = input[channel_in_group + i*group_size + group*group_size*h*w] - mean_val;
            var_sum += x*x;
        }}
        float var_val = var_sum / (h*w) + eps;
        
        // Normalize and scale/shift
        float inv_std = 1.0 / sqrt(var_val);
        for(int i=0; i<h*w; i++) {{
            int idx = channel_in_group + i*group_size + group*group_size*h*w;
            output[idx] = (input[idx] - mean_val) * inv_std * weight[group] + bias[group];
        }}
    }}
    """
    
    cuda_src = f"""
    {kernel_code}
    
    extern "C" void launch_groupnorm_forward(cudaStream_t stream, 
                                             float* input, float* weight, float* bias,
                                             float* output, float* mean, float* var,
                                             int num_groups, int group_size, int n, int h, int w, float eps) {{
        const int block_size = 256;
        const int num_blocks = (n*h*w*num_groups*group_size + block_size - 1) / block_size;
        groupnorm_forward<<<num_blocks, block_size, 0, stream>>>(
            input, weight, bias, output, mean, var, 
            num_groups, group_size, n, h, w, eps
        );
    }}
    """
    
    load_inline(name="groupnorm_forward", 
               cuda_sources=cuda_src,
               functions=["launch_groupnorm_forward"],
               verbose=False)
    
    # Launch kernel
    module = load_inline(...)
    module.launch_groupnorm_forward(stream, 
                                   input.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                                   output.data_ptr(), mean.data_ptr(), var.data_ptr(),
                                   num_groups, group_size, n, h, w, eps)
    
def load_groupnorm_backward(stream, grad_output, input, weight, bias, mean, var, 
                           grad_input, grad_weight, grad_bias, num_groups, group_size, n, h, w, eps):
    # CUDA kernel code for backward pass
    kernel_code = f"""
    extern "C" __global__ void groupnorm_backward(float* grad_output, float* input, float* weight, 
                                                  float* bias, float* mean, float* var,
                                                  float* grad_input, float* grad_weight, float* grad_bias,
                                                  int num_groups, int group_size, int n, int h, int w, float eps) {{
        // Implementation similar to forward but for gradient computation
    }}
    """
    
    cuda_src = f"""
    {kernel_code}
    
    extern "C" void launch_groupnorm_backward(cudaStream_t stream, 
                                              float* grad_out, float* input, float* weight, float* bias,
                                              float* mean, float* var, 
                                              float* grad_in, float* grad_w, float* grad_b,
                                              int num_groups, int group_size, int n, int h, int w, float eps) {{
        // Kernel launch configuration
    }}
    """
    
    module = load_inline(name="groupnorm_backward", 
                        cuda_sources=cuda_src,
                        functions=["launch_groupnorm_backward"],
                        verbose=False)
    
    module.launch_groupnorm_backward(...)
    
class GroupNormCUDA(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(GroupNormCUDA, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x):
        return GroupNormCUDAFunction.apply(x, self.weight, self.bias, self.num_groups, self.eps)
    
class ModelNew(nn.Module):
    def __init__(self, num_features, num_groups):
        super(ModelNew, self).__init__()
        self.gn = GroupNormCUDA(num_groups=num_groups, num_channels=num_features)
        
    def forward(self, x):
        return self.gn(x)