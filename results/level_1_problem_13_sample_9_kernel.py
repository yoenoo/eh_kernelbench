cuda
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    int bx = blockIdx.x; // block index x
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices for tiles in the output matrix:
    int blockRow = by * TS;
    int blockCol = bx * TS;
    int row = blockRow + ty;
    int col = blockCol + tx;

    // Each thread computes a single element of C. Accumulator
    float Cval = 0.0f;

    // Load tiles into shared memory
    __shared__ float ds_A[TS][TS];
    __shared__ float ds_B[TS][TS];

    for (int m = 0; m < (N + TS -1)/TS; ++m) {
        // Load the current tile of A and B into shared memory

        int aRow = blockRow + ty;
        int aCol = m * TS + tx;
        if (aRow < N && aCol < N) {
            ds_A[ty][tx] = A[aRow * N + aCol];
        }
        else {
            ds_A[ty][tx] = 0.0f;
        }

        int bRow = m * TS + ty;
        int bCol = blockCol + tx;
        if (bRow < N && bCol < N) {
            ds_B[ty][tx] = B[bRow * N + bCol];
        }
        else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the partial sum
        for (int k = 0; k < TS; ++k) {
            Cval += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    // Write the computed value to C
    if (row < N && col < N) {
        C[row * N + col] = Cval;
    }
}