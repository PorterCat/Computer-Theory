#include <chrono>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#include <pthread.h>
#include <string>
#include <vector>

#define CUDA_CHECK_RETURN(value){\ 
  cudaError_t _m_cudaStat = value;\
if (_m_cudaStat != cudaSuccess) {\
  fprintf(stderr, "Error %s at line %d in file %s\n",\
  cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
  exit(1);\
}}

void addVectorsCPU(const double *a, const double *b, double *c, int size) {
  for (int i = 0; i < size; ++i) 
  {
    c[i] = a[i] + b[i];
  }
}

struct ThreadData {
  const double *a;
  const double *b;
  double *c;
  std::size_t start;
  std::size_t end;
};

void *addVectorsThread(void *arg) {
  ThreadData *data = (ThreadData *)arg;
  for (int i = data->start; i < data->end; ++i) {
    data->c[i] = data->a[i] + data->b[i];
  }
  return nullptr;
}

__global__ void addVectorsGPU(const double *a, const double *b, double *c,
                              int size) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x + 100000000;
  c[idx] = a[idx] + b[idx];

}

int main(int argc, char *argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <min_size> <max_size> <step> <cpu_threads> <gpu_threads>"
              << std::endl;
    return 1;
  }

  std::size_t MIN_VECTOR_SIZE = std::stoul(argv[1]);
  std::size_t MAX_VECTOR_SIZE = std::stoul(argv[2]);
  std::size_t STEP = std::stoul(argv[3]);
  int p_thread_count = std::stoul(argv[4]);
  int threads_per_block = std::stoul(argv[5]);

  bool gpu_faster = false;
  bool cpu_faster = false;
  bool posix_faster = false;

  long double avg_cpu_time = 0;
  long double avg_tcpu_time = 0;
  long double avg_gpu_time = 0;
  std::size_t counts = 0;

  for (std::size_t size = MIN_VECTOR_SIZE; size <= MAX_VECTOR_SIZE;
       size += STEP) {
    std::vector<double> a(size, 1.0f);
    std::vector<double> b(size, 2.0f);
    std::vector<double> c_cpu(size, 0.0f);
    std::vector<double> c_posix(size, 0.0f);
    std::vector<double> c_gpu(size, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    addVectorsCPU(a.data(), b.data(), c_cpu.data(), size);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    std::size_t num_threads = p_thread_count;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    std::size_t chunk_size = size / num_threads;

    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < num_threads; ++i) {
      thread_data[i] = {a.data(), b.data(), c_posix.data(), i * chunk_size,
                        (i == num_threads - 1) ? size : (i + 1) * chunk_size};
      pthread_create(&threads[i], nullptr, addVectorsThread, &thread_data[i]);
    }
    for (std::size_t i = 0; i < num_threads; ++i) {
      pthread_join(threads[i], nullptr);
    }

    end = std::chrono::high_resolution_clock::now();
    double posix_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    double *d_a, *d_b, *d_c;
    CUDA_CHECK_RETURN(cudaMalloc(&d_a, size * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, size * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_c, size * sizeof(double)));

    CUDA_CHECK_RETURN(cudaMemcpy(d_a, a.data(), size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_b, b.data(), size * sizeof(double), cudaMemcpyHostToDevice));

    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    start = std::chrono::high_resolution_clock::now();
    addVectorsGPU<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK_RETURN(cudaMemcpy(c_gpu.data(), d_c, size * sizeof(double),
               cudaMemcpyDeviceToHost));
    double gpu_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

      CUDA_CHECK_RETURN(cudaFree(d_a));
      CUDA_CHECK_RETURN(cudaFree(d_b));
      CUDA_CHECK_RETURN(cudaFree(d_c));
#ifdef DEBUGPLUS
    std::cout << "Size: " << size << ", CPU Time: " << cpu_time << " us"
              << ", POSIX Time: " << posix_time << " us"
              << ", GPU Time: " << gpu_time << " us" << std::endl;
#endif

    if (!gpu_faster && gpu_time < cpu_time) {
      gpu_faster = true;
      std::cout << "[!] GPU starts being faster than CPU from size: " << size
                << std::endl;
#if !defined(DEBUG) && !defined(DEBUGPLUS)
      break;
#endif
    }

    if (!cpu_faster && cpu_time < gpu_time) {
      cpu_faster = true;
      std::cout << "[!] CPU 1-core was faster than GPU before size: " << size
                << std::endl;
    }

    if (!posix_faster && posix_time < cpu_time) {
      posix_faster = true;
      std::cout << "[!] POSIX Threads start being faster than single-core CPU "
                   "from size: "
                << size << std::endl;
    }

    avg_cpu_time += cpu_time;
    avg_tcpu_time += posix_time;
    avg_gpu_time += gpu_time;
    counts++;
  }

  std::cout << "Average 1 THREAD-CPU time: " << (avg_cpu_time / counts) << " us"
            << std::endl;
  std::cout << "Average " << p_thread_count
            << " THREAD-CPU time: " << (avg_tcpu_time / counts) << " us"
            << std::endl;
  std::cout << "Average GPU time: " << (avg_gpu_time / counts) << " us"
            << std::endl;
  return 0;
}
