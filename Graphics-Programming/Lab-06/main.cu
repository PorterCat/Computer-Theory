#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define TILE_DIM 32

__global__ void transposeNaive(float *odata, const float *idata, int width, int height) 
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    odata[x * height + y] = idata[y * width + x];
}

__global__ void transposeShared(float *odata, const float *idata, int width, int height) 
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transposeSharedNoBankConflicts(float *odata, const float *idata, int width, int height) 
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

void transposeOnCPU(float *odata, const float *idata, int width, int height)
{    
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            odata[j * height + i] = idata[i * width + j];
        }
    }
}

int main() 
{
    int width = 1024;
    int height = 1024;

    printf("Size: %d x %d\n", width, height);
    size_t size = width * height * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < width * height; ++i) 
        h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto startCpu = std::chrono::high_resolution_clock::now();
    transposeOnCPU(h_out, h_in, width, height);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = endCpu - startCpu;
    printf("Transpose on CPU execution time: %f ms\n", duration.count() * 1000);


    #ifndef Native
    cudaEventRecord(start, 0);
    transposeNaive<<<grid, threads>>>(d_out, d_in, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Transpose Naive execution time: %f ms\n", elapsedTime);
    #endif

    #ifndef SharedConflict
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    transposeShared<<<grid, threads>>>(d_out, d_in, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Transpose Shared (with conflicts) execution time: %f ms\n", elapsedTime);
    #endif

    #ifndef SharedNoConflict
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    transposeSharedNoBankConflicts<<<grid, threads>>>(d_out, d_in, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Transpose Shared (no bank conflicts) execution time: %f ms\n", elapsedTime);
    #endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    
    delete [] h_in;
    delete [] h_out;

    return 0;
}
