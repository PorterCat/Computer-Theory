#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <iostream>

// #define GET_BANK(arr, index) ({ \
//     uintptr_t base_address = reinterpret_cast<uintptr_t>(arr); \
//     uintptr_t element_address = reinterpret_cast<uintptr_t>(&arr[index]); \
//     size_t offset = (element_address - base_address) / sizeof(arr[0]); \
//     (offset % 32); \
// })

// __global__ void test()
// {
//     __shared__  int data[64];
//     int index = threadIdx.x * 2;

//     size_t bank = GET_BANK(data, index);

//     printf("Thread %d: Bank %d\n", index, bank);

//     index = threadIdx.x * 2 + 1;

//     bank = GET_BANK(data, index);

//     printf("Thread %d: Bank %d\n", index, bank);
    
// }

__global__ void test()
{
    __shared__ int shared[64];

    int a = shared[threadIdx.x];
}

int main()
{
    test<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}