#ifndef MODULE8_HELPERFUNCTIONS_CUH
#define MODULE8_HELPERFUNCTIONS_CUH

#include <cstdio>
#include <cassert>
#include <vector>
#include <cublas.h>
#include <cusolverDn.h>
#include <cassert>
#include <cublas.h>
#include <chrono>
#include <random>
#include <iostream>

// Constants
#define MAX 5
#define MIN -5

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

inline std::mt19937 &generator()
{
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(clock());
    return gen;
}

template<typename T>
void freeDeviceAlloc(T var)
{
    if (var)
    {
        cudaFree(var);
    }
}

template<typename T, typename ... Types>
void freeDeviceAlloc(T var1, Types... vars)
{
    if(var1)
    {
        cudaFree(var1);
    }
    freeDeviceAlloc(vars...);
}

template<typename T>
void freeHostAlloc(T var)
{
    if (var)
    {
        free(var);
    }
}

template<typename T, typename ... Types>
void freeHostAlloc(T var1, Types ... vars)
{
    if(var1)
    {
        free(var1);
    }
    freeHostAlloc(vars...);
}

template<typename T>
void freePinned(T var1)
{
    if(var1)
    {
        cudaFreeHost(var1);
    }
}

template<typename T, typename ... Types>
void freePinned(T var1, Types ... vars)
{
    if(var1)
    {
        cudaFreeHost(var1);
    }
    freePinned(vars...);
}

// A function to generate integers in the range [min, max]
template<typename T>
T randEngine()
{
    std::uniform_real_distribution<T> dist(MIN, MAX);
    return dist(generator());
}

template<typename T>
void
printArray(T *array, const unsigned int numRows, const unsigned int numCols)
{
    for (auto i = 0; i < numRows; ++i)
    {
        for (auto j = 0; j < numCols; ++j)
        {
            printf("%03.6f ", array[j * numRows + i]);
        }
        std::cout << std::endl;
    }
}


void fillArrays(float *a, const int rows, const int cols);

void printMultiplicationResults(float *deviceA, float *deviceB, float *deviceC,
                                const int aRows, const int aCols,
                                const int bRows, const int bCols);

void generateSymmetricMatrix(const int size, double *matrix);

void
gpuArrayMult(const float *a, const float *b, float *c, const int m, const int n,
             const int k);

void findSymmetricMatrixEigenValues(double *d_A, double *d_W, const int m);

void runMatrixMult(float *a, float *b, const int aRows, const int aCols,
                   const int bRows, const int bCols);
void
runEigen(double *matrix, const int eigenDim);

void computeEigenvalues(double *matrix, const int dim);

void printEigenResults(double *array, double *eigenVals, const int dim);

#endif //MODULE8_HELPERFUNCTIONS_CUH
