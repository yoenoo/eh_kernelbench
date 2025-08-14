import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Compile the custom CUDA kernel for inclusive scan
        self.inclusive_scan = load_inline(
            name="inclusive_scan",
            cpp_sources=f"""
                torch::Tensor inclusive_scan_cuda(torch::Tensor input, int64_t dim);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda.h>
                #include <cuda_runtime.h>
                #include <cub/cub.cuh>

                template <typename T>
                __global__ void inclusive_scan_kernel(void *d_temp_storage, size_t *temp_storage_bytes, T *out, const T *in, int volume, int dim_size, int dim, int dim_offset, int other_dims) {{
                    using BlockReduce = cub::BlockReduce<T, 256>;
                    void *q = d_temp_storage;
                    size_t required = 0;
                    if (dim_offset < other_dims) {{
                        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
                        if (idx == 0) {{
                            cub::DeviceScan::InclusiveScan(nullptr, required, in, out, cub::Sum(), dim_size);
                            *temp_storage_bytes = required;
                        }}
                        __syncthreads();
                        if (threadIdx.x < required) {{
                            ((char*)q)[threadIdx.x] = 0;
                        }}
                        cub::DeviceScan::InclusiveScan((void*)q, required, in, out, cub::Sum(), dim_size);
                    }}
                    else {{
                        auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (global_idx < volume) {{
                            auto dim_idx = global_idx % dim_size;
                            T val = in[global_idx];
                            T prev = (dim_idx == 0) ? T(0) : out[global_idx - 1];
                            out[global_idx] = prev + val;
                        }}
                    }}
                }}

                torch::Tensor inclusive_scan_cuda(torch::Tensor input, int64_t dim) {{
                    auto input_shape = input.sizes();
                    int dim_size = input.size(dim);
                    int other_dims = 1;
                    for(int i=0; i<input.dim(); i++) {{
                        if(i != dim) other_dims *= input.size(i);
                    }}
                    auto volume = input.numel();
                    auto dtype = input.dtype();
                    auto output = torch::empty_like(input);

                    dim3 block(256);
                    dim3 grid(1);

                    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
                    void *d_temp_storage = nullptr;
                    size_t temp_storage_bytes = 0;

                    auto func = ([=] __device__ (T a, T b) {{ return a + b; }});

                    auto in_data = input.data_ptr<float>();
                    auto out_data = output.data_ptr<float>();

                    int blocks_per_feature = (other_dims + grid.x - 1)/grid.x;
                    dim3 grid(blocks_per_feature);

                    inclusive_scan_kernel<float>
                        <<<grid, block, 0, stream.stream()>>>
                        (d_temp_storage, &temp_storage_bytes,
                        out_data, in_data, volume, dim_size, dim, blockIdx.x, other_dims);

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {{
                        printf("Error in inclusive_scan_kernel: %s\\n", cudaGetErrorString(err));
                    }}
                    return output;
                }}
            """,
            functions=["inclusive_scan_cuda"],
            verbose=True,
            extra_cflags=["-x", "cu"],
            extra_cuda_cflags=["-arch=sm_75"],
        )

    def forward(self, x):
        return self.inclusive_scan.inclusive_scan_cuda(x, self.dim)