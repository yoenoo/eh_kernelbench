__global__ void conv_transpose1d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels,
    int input_length,
    int kernel_size,
    int dilation,
    int output_length
)