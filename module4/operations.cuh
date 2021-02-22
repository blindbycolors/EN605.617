//
// Created by nou on 2/21/21.
//

#include "string"
#include "assert.h"

#ifndef MODULE4_OPERATION_CUH
#define MODULE4_OPERATION_CUH

#define MIN 1
#define MAX 3
#define MIN_PRINTABLE 32
#define MAX_PRINTABLE 126
#define MAX_ALPHA 94

#define HOST_ALLOCATE_PAGEABLE(total, a, type)                   \
    type* a = (type *) malloc(sizeof(type) * total);

#define HOST_ALLOCATE_PINNED(total, a, type)                   \
    type * a;                                                           \
    checkCuda(cudaMallocHost((void**) &a, sizeof(type) * total));

#define DO_OPERATIONS(a, b, function, duration, opType) \
    auto duration = function(numBlocks, blockSize, totalThreads, a, b);\
    printf("%s\n", opType) ;\
    printf("\tDuration: %f\n\tEffective bandwidth (GB/s): %f\n",\
    duration, totalThreads*4*3/duration/1e6);

#define FREE_PINNED(a, ...) \
    cudaFreeHost(a);        \
    _FREE_PINNED(__VA_ARGS__);

#define _FREE_PINNED(a, ...) \
    cudaFreeHost(a);

#define FREE_PAGEABLE(a, ...) \
    free(a);\
    _FREE_PAGEABLE(__VA_ARGS__);

#define  _FREE_PAGEABLE(a, ...) \
    free(a);

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

// Kernel Functions
__global__ void kernel_addition(const int*, const int*, float*);
__global__ void kernel_subtraction(const int*, const int*, float*);
__global__ void kernel_multiplication(const int*, const int*, float*);
__global__ void kernel_modulus(const int*, const int*, float*);
__global__ void kernel_caeser_encrypt(const char *, int, int, char *);
__global__ void kernel_caeser_decrypt(const char *, int, int, char *);

// Functions to handle copying memory from host to device
float hostAdd( int, int, int, int*, int*);
float hostSubtract(int, int, int, int*, int*);
float hostMultiply(int, int, int, int*, int*);
float hostMod(int, int, int, int*, int*);
std::string hostEncrypt(int, int, const char*, int, int);
std::string hostDecrypt(int, int, const char*, int, int);

#endif //MODULE4_OPERATION_CUH
