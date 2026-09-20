// matmul.cu
// HW1 CUDA task: tiled matrix multiplication with CUDA C, benchmarked against a CPU baseline.
//
// SID4=3215 SEED=3215 (personal parameters; not used numerically by this kernel, recorded per
// the standing assignment requirements).
//
// Usage:
//   nvcc -O3 -arch=sm_75 matmul.cu -o matmul
//   ./matmul <N>          // N = square matrix dimension, e.g. 256, 1024, 4096
//
// Design notes on blocks/threads (see inline comments in matMulTiled for the full explanation):
//   - Each CUDA thread computes exactly one output element C[row][col].
//   - Threads are grouped into 2D thread blocks of size TILE x TILE (16x16 = 256 threads/block).
//   - The grid is a 2D array of blocks sized so that gridDim * blockDim covers the whole N x N
//     output matrix, i.e. grid = ceil(N/TILE) x ceil(N/TILE) blocks.
//   - Each block cooperatively loads TILE x TILE tiles of A and B into fast on-chip shared memory,
//     reused by all TILE*TILE threads in the block before advancing to the next tile along K.
//     This tiling cuts global-memory traffic by a factor of ~TILE compared to a naive kernel where
//     every thread re-reads full rows/columns of A and B from global memory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// -----------------------------------------------------------------------------------------------
// Tiled GPU matrix multiplication kernel: C = A * B, all matrices N x N, row-major.
//
// Blocks and threads:
//   - blockDim = (TILE, TILE): a 2D block of TILE*TILE = 256 threads. Thread (tx, ty) within a
//     block is responsible for one element of the TILE x TILE output tile owned by this block.
//   - gridDim = (ceil(N/TILE), ceil(N/TILE)): enough 2D blocks to cover every TILE x TILE tile of
//     the N x N output matrix C. blockIdx.x/blockIdx.y select which output tile this block owns.
//   - Global row/col for this thread: row = blockIdx.y*TILE + threadIdx.y,
//                                      col = blockIdx.x*TILE + threadIdx.x.
//   - Parallelism: all N*N output elements are computed concurrently by different threads (subject
//     to how many can be resident on the SMs at once); within a block, threads cooperate via
//     __shared__ memory tiles As/Bs, loading one TILE x TILE tile of A and one of B per iteration
//     of the outer loop over the K dimension, syncing with __syncthreads() so every thread's
//     shared-memory reads happen after all loads finish (and before the next iteration overwrites
//     the tile).
// -----------------------------------------------------------------------------------------------
__global__ void matMulTiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;   // output row this thread computes
    int col = blockIdx.x * TILE + tx;   // output column this thread computes

    float acc = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();  // wait until the whole tile is loaded before reading it

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // wait until all threads finish reading before next tile overwrites it
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

// -----------------------------------------------------------------------------------------------
// CPU baseline: naive triple-loop matmul (single-threaded), used as the timing reference.
// -----------------------------------------------------------------------------------------------
void matMulCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void fillRandom(float* mat, int N, unsigned seed) {
    srand(seed);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

double maxAbsDiff(const float* a, const float* b, int N) {
    double maxDiff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double d = fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > maxDiff) maxDiff = d;
    }
    return maxDiff;
}

int main(int argc, char** argv) {
    const unsigned SEED = 3215;  // SID4, for reproducible random inputs
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    printf("Matrix size: %d x %d (TILE=%d)\n", N, N, TILE);

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    float* h_A = static_cast<float*>(malloc(bytes));
    float* h_B = static_cast<float*>(malloc(bytes));
    float* h_C_gpu = static_cast<float*>(malloc(bytes));
    float* h_C_cpu = static_cast<float*>(malloc(bytes));

    fillRandom(h_A, N, SEED);
    fillRandom(h_B, N, SEED + 1);

    // ---------------- CPU baseline timing ----------------
    // Skip the full O(N^3) CPU run for N=4096 (would take a very long time single-threaded);
    // still verify correctness on smaller sizes.
    double cpuMs = -1.0;
    bool ranCpu = (N <= 2048);
    if (ranCpu) {
        auto cpuStart = std::chrono::high_resolution_clock::now();
        matMulCPU(h_A, h_B, h_C_cpu, N);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
        printf("CPU time:            %10.3f ms\n", cpuMs);
    } else {
        printf("CPU time:            skipped (N=%d too large for naive single-thread CPU loop)\n", N);
    }

    // ---------------- GPU setup ----------------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t evH2DStart, evH2DEnd, evKernelStart, evKernelEnd, evD2HStart, evD2HEnd;
    CUDA_CHECK(cudaEventCreate(&evH2DStart));
    CUDA_CHECK(cudaEventCreate(&evH2DEnd));
    CUDA_CHECK(cudaEventCreate(&evKernelStart));
    CUDA_CHECK(cudaEventCreate(&evKernelEnd));
    CUDA_CHECK(cudaEventCreate(&evD2HStart));
    CUDA_CHECK(cudaEventCreate(&evD2HEnd));

    // ---------------- H2D transfer ----------------
    CUDA_CHECK(cudaEventRecord(evH2DStart));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(evH2DEnd));

    // ---------------- Kernel launch ----------------
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    printf("Grid: (%d, %d) blocks, Block: (%d, %d) threads => %d threads total\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y,
           gridDim.x * gridDim.y * blockDim.x * blockDim.y);

    // Warm-up launch (excluded from timing) so the timed run isn't skewed by first-launch overhead.
    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(evKernelStart));
    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(evKernelEnd));
    CUDA_CHECK(cudaEventSynchronize(evKernelEnd));
    CUDA_CHECK(cudaGetLastError());

    // ---------------- D2H transfer ----------------
    CUDA_CHECK(cudaEventRecord(evD2HStart));
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evD2HEnd));
    CUDA_CHECK(cudaEventSynchronize(evD2HEnd));

    float h2dMs = 0, kernelMs = 0, d2hMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2dMs, evH2DStart, evH2DEnd));
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, evKernelStart, evKernelEnd));
    CUDA_CHECK(cudaEventElapsedTime(&d2hMs, evD2HStart, evD2HEnd));
    float transferMs = h2dMs + d2hMs;
    float endToEndMs = h2dMs + kernelMs + d2hMs;

    printf("GPU H2D transfer:    %10.3f ms\n", h2dMs);
    printf("GPU kernel time:     %10.3f ms\n", kernelMs);
    printf("GPU D2H transfer:    %10.3f ms\n", d2hMs);
    printf("GPU H2D+D2H total:   %10.3f ms\n", transferMs);
    printf("GPU end-to-end time: %10.3f ms\n", endToEndMs);

    if (ranCpu) {
        double speedupKernelOnly = cpuMs / kernelMs;
        double speedupEndToEnd = cpuMs / endToEndMs;
        printf("Speedup (CPU / GPU kernel only):   %8.2fx\n", speedupKernelOnly);
        printf("Speedup (CPU / GPU end-to-end):    %8.2fx\n", speedupEndToEnd);

        double diff = maxAbsDiff(h_C_gpu, h_C_cpu, N);
        printf("Max abs diff GPU vs CPU: %.6e %s\n", diff, (diff < 1e-2) ? "(OK)" : "(CHECK!)");
    }

    // ---------------- Cleanup ----------------
    CUDA_CHECK(cudaEventDestroy(evH2DStart));
    CUDA_CHECK(cudaEventDestroy(evH2DEnd));
    CUDA_CHECK(cudaEventDestroy(evKernelStart));
    CUDA_CHECK(cudaEventDestroy(evKernelEnd));
    CUDA_CHECK(cudaEventDestroy(evD2HStart));
    CUDA_CHECK(cudaEventDestroy(evD2HEnd));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return 0;
}
