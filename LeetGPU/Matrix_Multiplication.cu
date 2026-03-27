// https://leetgpu.com/challenges/matrix-multiplication

#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void matrix_multiplication_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Declare Shared Memory to store the sub-matrices (Tiles) of A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Calculate which row and col of matrix C the current thread is responsible for
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Divide the dot product process into multiple Tiles to perform
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // 1. Collaboratively load data from Global Memory to Shared Memory
        // Note the boundaries check: pad with 0 if it exceeds the matrix bounds
        if (row < M && (t * TILE_SIZE + tx) < N)
            s_A[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        else
            s_A[ty][tx] = 0.0f;

        if ((t * TILE_SIZE + ty) < N && col < K)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * K + col];
        else
            s_B[ty][tx] = 0.0f;

        // Wait for all threads in the Block to finish loading
        __syncthreads();

        // 2. Perform local matrix multiplication inside the ultra-fast Shared Memory
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += s_A[ty][i] * s_B[i][tx];
        }

        // Wait for all threads to finish computing before entering the next loop to overwrite Shared Memory
        __syncthreads();
    }

    // 3. Write the final result back to Global Memory
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiplication_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}