torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride, // assuming 1D stride for simplicity (same in all dimensions)
    int padding,
    int dilation
);