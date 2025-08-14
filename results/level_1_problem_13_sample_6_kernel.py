cuda
__global__ void MatrixMulKernel(float* C, float* A, float* B, int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_WIDTH;
    int by = blockIdx.y * TILE_WIDTH;

    float Cvalue = 0;

    for (int m = 0; m < (N / TILE_WIDTH); ++m)
    {
        // Sub-matrix of A starting at row by, column m*TILE_WIDTH
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        // Load the tiles into shared memory
        As[ty][tx] = A[by * N + (m*TILE_WIDTH + tx)];
        Bs[ty][tx] = B[(m*TILE_WIDTH + ty) * N + bx + tx];

        __syncthreads();

        // Compute the product of the two tiles
        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    // Write the block sub-matrix to device memory
    int row = by + ty;
    int col = bx + tx;
    C[row * N + col] = Cvalue;
}