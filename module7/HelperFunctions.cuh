#include <cstdio>
#include <cassert>
#include <vector>

#ifndef MODULE7_HELPERFUNCTIONS_CUH
#define MODULE7_HELPERFUNCTIONS_CUH

// Constants
#define MIN 1
#define MAX 3

// Helper macros
#define ALLOCATE_PAGEABLE_MEMORY(a, type, total)        \
    type* a = (type *) malloc(sizeof(type) * total);    \
    // End ALLOCATE_PAGEABLE_MEMORY

#define ALLOCATE_PINNED_MEMORY(a, type, total)                     \
    type * a;                                                      \
    checkCuda(cudaMallocHost((void**) &a, sizeof(type) * total));  \
    // End ALLOCATE_PINNED_MEMORY

#define DEVICE_ALLOCATE(a, type, totalThreads)                          \
    type *a;                                                            \
    checkCuda(cudaMalloc((void**) &a, sizeof(type) * totalThreads));    \
    // End DEVICE_ALLOCATE macro

#define ACTIVATE_KERNEL(MATH_OP)                                              \
    auto blockSize = numStreams / numBlocks;                                  \
    cudaEvent_t start, stop;                                                  \
    cudaStream_t stream;                                                      \
    auto delta = 0.0F;                                                        \
                                                                              \
    checkCuda(cudaEventCreate(&start));                                       \
    checkCuda(cudaEventCreate(&stop));                                        \
    checkCuda(cudaStreamCreate(&stream));                                     \
                                                                              \
    checkCuda(cudaEventRecord(start, 0));                                     \
    checkCuda(cudaMemcpyAsync(d_a, h_a, numStreams * sizeof(int),             \
        cudaMemcpyHostToDevice, stream));                                     \
    checkCuda(cudaMemcpyAsync(d_b, h_b, numStreams * sizeof(int),             \
        cudaMemcpyHostToDevice, stream));                                     \
    MATH_OP<<<numBlocks, blockSize, 1, stream>>>(d_a, d_b, d_out);            \
    checkCuda(cudaMemcpyAsync(h_out, d_out, numStreams * sizeof(int),         \
        cudaMemcpyDeviceToHost, stream));                                     \
    checkCuda(cudaStreamSynchronize(stream));                                 \
    checkCuda(cudaEventRecord(stop, 0));                                      \
    checkCuda(cudaEventSynchronize(stop));                                    \
    checkCuda(cudaEventElapsedTime(&delta, start, stop));                     \
    // End ACTIVATE_KERNEL

// Inline functions / Utility Methods
// Convenience function for checking CUDA runtime API results
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

template<typename T>
void freeDeviceAlloc(T var)
{
    cudaFree(var);
}

template<typename T, typename ... Types>
void freeDeviceAlloc(T var1, Types... vars)
{
    cudaFree(var1);
    freeDeviceAlloc(vars...);
}

template<typename T>
void freeHostAlloc(T var)
{
    free(var);
}

template<typename T, typename ... Types>
void freeHostAlloc(T var1, Types ... vars)
{
    free(var1);
    freeHostAlloc(vars...);
}

template<typename T>
void freePinned(T var1)
{
    cudaFreeHost(var1);
}

template<typename T, typename ... Types>
void freePinned(T var1, Types ... vars)
{
    cudaFreeHost(var1);
    freePinned(vars...);
}

// Kernel functions
__global__ void kernelAdd(const int *a, const int *b, int *out);
__global__ void kernelSubtract(const int *a, const int *b, int *out);
__global__ void kernelMultiply(const int *a, const int *b, int *out);
__global__ void kernelModulus(const int *a, const int * b, int *out);

// Wrapper functions
void hostAdd(const int *h_a, const int *h_b, int numStreams,
             int numBlocks, std::vector<float> &time);
void hostSub(const int *h_a, const int *h_b, int numStreams,
             int numBlocks, std::vector<float> &time);
void hostMult(const int *h_a, const int *h_b, int numStreams,
              int numBlocks, std::vector<float> &time);
void hostMod(const int *h_a, const int *h_B, int numStreams,
             int numBlocks, std::vector<float> &time);


#endif //MODULE7_HELPERFUNCTIONS_CUH
