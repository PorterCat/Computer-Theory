#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#define N 1000000  // Размер вектора

__global__ void vectorAdd(const float *A, const float *B, float *C, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

float runKernel(long threadsPerBlock) {
    long blocksPerGrid = N / threadsPerBlock;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_A.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return milliseconds;
}

int main() {
    std::ofstream outFile("result.csv");

    for (long T = 16; T <= 512; T += 16) {
        float time = runKernel(T);
        outFile << T << " " << time << "\n";
    }

    outFile.close();

    return 0;
}

