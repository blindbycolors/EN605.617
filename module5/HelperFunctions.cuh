#include <string>
#include <cassert>

#ifndef MODULE5_HELPERFUNCTIONS_CUH
#define MODULE5_HELPERFUNCTIONS_CUH

#define MIN 1
#define MAX 3
#define NUM_ELEMENTS 1024
#define MIN_PRINTABLE 32
#define MAX_PRINTABLE 126
#define MAX_ASCII 128

#define HOST_ALLOCATE_PAGEABLE(a, type, total)                                 \
    type* a = (type *) malloc(sizeof(type) * total);                           \
    // End HOST_ALLOCATE_PAGEABLE

#define HOST_ALLOCATE_PINNED(a, type, total)                                   \
    type * a;                                                                  \
    checkCuda(cudaMallocHost((void**) &a, sizeof(type) * total));              \
    // End HOST_ALLOCATE_PINNED

#define ACTIVATE_SHARED_OP_KERNEL(opFunc, d_a, d_b, d_out, kernelStart,        \
        kernelStop, delta)                                                     \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart,0));                                 \
    opFunc<<<1,NUM_ELEMENTS>>>(d_a, d_b, d_out);                               \
    checkCuda(cudaEventRecord(kernelStop,0));                                  \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));                                   \
    printf("\tDuration %f\n\tEffective bandwidth (GB/s): %f\n", delta,         \
        NUM_ELEMENTS*4*3/delta/1e6);                                           \
    // End ACTIVATE_SHARED_OP_KERNEL macro

#define ACTIVATE_CONSTANT_OP_KERNEL(opFunc, d_out, h_a, h_b,            \
        kernelStart, kernelStop, delta)                                        \
    cudaEvent_t kernelStart, kernelStop;                                       \
    cudaMemcpyToSymbol(constantArrA, h_a, sizeof(int) * NUM_ELEMENTS);         \
    cudaMemcpyToSymbol(constantArrB, h_b, sizeof(int) * NUM_ELEMENTS);         \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart,0));                                 \
    opFunc<<<1,NUM_ELEMENTS>>>(d_out);                                         \
    checkCuda(cudaEventRecord(kernelStop,0));                                  \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));                                   \
    printf("\tDuration %f\n\tEffective bandwidth (GB/s): %f\n", delta,         \
        NUM_ELEMENTS*4*3/delta/1e6);                                           \
    // End ACTIVATE_CONSTANT_OP_KERNEL

#define ACTIVATE_SHARED_CAESER_KERNEL(opFunction, d_a, d_out, offset)                 \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart, 0));                                \
    opFunction<<<1, NUM_ELEMENTS>>>(d_a, offset, d_out);                       \
    checkCuda(cudaEventRecord(kernelStop, 0));                                 \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));                                   \
    printf("\tDuration %f\n\tEffective bandwidth (GB/s): %f\n", delta,         \
        NUM_ELEMENTS*4*3/delta/1e6);                                           \
    // End ACTIVATE_SHARED_CAESER_KERNEL macro

#define ACTIVATE_CONSTANT_CAESER_KERNEL(opFunction, d_out, h_a, offset) \
    cudaMemcpyToSymbol(constantInput, h_a, sizeof(char) * NUM_ELEMENTS);  \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart, 0));                                \
    opFunction<<<1, NUM_ELEMENTS>>>(offset, d_out);                       \
    checkCuda(cudaEventRecord(kernelStop, 0));                                 \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));                                   \
    printf("\tDuration %f\n\tEffective bandwidth (GB/s): %f\n", delta,         \
        NUM_ELEMENTS*4*3/delta/1e6);                                           \
    // End ACTIVATE_SHARED_CAESER_KERNEL macro

#define DEVICE_ALLOCATE(a, type, totalThreads)                                 \
    type *a;                                                                   \
    checkCuda(cudaMalloc((void**) &a, sizeof(type) * totalThreads));           \
    // End DEVICE_ALLOCATE macro

// Convenience function for checking CUDA runtime API results
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void sharedAdd(const int* a, const int* b, int *out);

__global__ void sharedSub(const int* a, const int* b, int *out);

__global__ void sharedMult(const int* a, const int* b, int *out);

__global__ void sharedMod(const int* a, const int* b, int *out);

__global__ void sharedCeaserEncrypt(const char *value,
                                    const int offset,
                                    char *d_out);

__global__ void sharedCeaserDecrypt(const char *value,
                                    const int offset,
                                    char *d_out);

// Utility Methods
template <typename T>
void freeDeviceAlloc(T var)
{
    cudaFree(var);
}

template <typename T, typename ... Types>
void freeDeviceAlloc(T var1, Types... vars)
{
    cudaFree(var1);
    freeDeviceAlloc(vars...);
}

template <typename T>
void freeHostAlloc(T var)
{
    free(var);
}

template <typename T, typename ... Types>
void freeHostAlloc(T var1, Types ... vars)
{
    free(var1);
    freeHostAlloc(vars...);
}

template <typename T>
void freePinned(T var1)
{
    cudaFreeHost(var1);
}

template <typename T, typename ... Types>
void freePinned(T var1, Types ... vars)
{
    cudaFreeHost(var1);
    freePinned(vars...);
}

void hostAdd(const int* h_a, const  int* h_b);

void hostSub(const int* h_a, const int* h_b);

void hostMult(const int* h_a, const int* h_b);

void hostMod(const int* h_a, const int* h_B);

std::string hostSharedEncrypt(const char *h_input, int offset);
std::string hostSharedDecrypt(const char *h_input, int offset);

std::string hostConstantEncrypt(const char *h_input, int offset);
std::string hostConstantDecrypt(const char *h_input, int offset);

#endif //MODULE5_HELPERFUNCTIONS_CUH
