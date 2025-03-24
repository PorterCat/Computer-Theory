#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorsSimple(float* A, float* B, int N, int K) // 3 4
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vectorIndex = idx % K;
    int vectorOffset = idx / K;
    int newPos = vectorIndex * N + vectorOffset;

    B[newPos] = A[idx];
}

__global__ void vectorsShared(float* A, float* B, int N, int K) 
{
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int vectorIndex = blockIdx.x;

    int bank = tid % 32;
    int row = tid / 32;
    int shared_idx = row + bank + 1;

    s_data[shared_idx] = A[vectorIndex * K + tid];
    __syncthreads();

    B[bank * N + vectorIndex] = s_data[shared_idx];
}

__global__ void vectorsRegPressure(float* A, float* B, int N, int K) 
{
    extern __shared__ float s_data[];
    float local_data[1000];

    int tid = threadIdx.x;
    int vectorIndex = blockIdx.x;

    for (int i = 0; i < 1000; ++i) local_data[i] = tid * i * 0.1f;

    s_data[tid] = A[vectorIndex * K + tid] + local_data[999];
    __syncthreads();

    B[tid * N + vectorIndex] = s_data[tid];
}

int main()
{
    // SET: 0 1 2 3 | 4 5 6 7 | 8 9 10 11   -> A*
    // GET: 0 4 8 | 1 5 9 | 2 6 10 | 3 7 11 -> B*
    
    int N = 128; // DEFAULT: 3
    int K = 64; // DEFAULT: 4

    size_t size = N * K * sizeof(float);

    float* h_A = new float[N * K];
    float* h_B = new float[N * K];

    for (int i = 0; i < N * K; ++i) 
    {
        h_A[i] = static_cast<float>(i);
    }

    float* d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // vectorsSimple<<<N, K>>>(d_A, d_B, N, K);
    vectorsShared<<<N, K, N * K * sizeof(float)>>>(d_A, d_B, N, K);
    //vectorsRegPressure<<<gridDim, blockDim>>>(d_A, d_B, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // std::cout << "Before: ";
    // for (int i = 0; i < K * N; ++i) std::cout << h_A[i] << " ";
    // std::cout << "\nAfter:  ";
    // for (int i = 0; i < K * N; ++i) std::cout << h_B[i] << " ";
    // std::cout << std::endl;

    delete[] h_A;
    delete[] h_B;

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}