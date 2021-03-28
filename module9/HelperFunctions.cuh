#ifndef MODULE9_THRUST_HELPERFUNCTIONS_CUH
#define MODULE9_THRUST_HELPERFUNCTIONS_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <random>

// Constants
#define MAX 100
#define MIN -100

// Helper macros
#define ALLOCATE_PAGEABLE_MEMORY(a, type, total)        \
    type* a = (type *) malloc(sizeof(type) * total);    \
    // End ALLOCATE_PAGEABLE_MEMORY

#define ALLOCATE_PINNED_MEMORY(a, type, total)                     \
    type * a;                                                      \
    checkCuda(cudaMallocHost((void**) &a, sizeof(type) * total));  \
    // End ALLOCATE_PINNED_MEMORY

#define DO_OPERATION(OP, DEVICE_ARRAY_NAME, HOST_ARRAY_NAME, LABEL)         \
    thrust::device_vector<int> DEVICE_ARRAY_NAME (dVectorA.size());         \
                                                                            \
    startTime = std::chrono::high_resolution_clock::now();                  \
    thrust::transform(dVectorA.begin(), dVectorA.end(), dVectorB.begin(),   \
        DEVICE_ARRAY_NAME.begin(), thrust::OP<int>());                      \
    endTime = std::chrono::high_resolution_clock::now();                    \
    duration = endTime - startTime;                                         \
    std::cout << "\t" << LABEL ;                                            \
    printf(" Runtime: %ld ns\n", std::chrono::duration_cast<                \
        std::chrono::nanoseconds>(duration).count());                       \
                                                                            \
    thrust::copy(DEVICE_ARRAY_NAME.begin(), DEVICE_ARRAY_NAME.end(),        \
        HOST_ARRAY_NAME.begin());                                           \
    // End DO_OPERATION macro

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


inline std::mt19937 &generator()
{
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(clock());
    return gen;
}


// A function to generate integers in the range [min, max]
inline int randEngine()
{
    std::uniform_int_distribution<int> dist(MIN, MAX);
    return dist(generator());
}

void printAll(int * hA, int * hB, int N,
              thrust::host_vector<int> add,
              thrust::host_vector<int> sub, thrust::host_vector<int> mul,
              thrust::host_vector<int> mod);

void print(thrust::host_vector<int> vec, const std::string &label);
void runOperations(int * hA,
                   int * hB,
                   int N,
                   thrust::host_vector<int> &hAdd,
                   thrust::host_vector<int> &hSub,
                   thrust::host_vector<int> &hMul,
                   thrust::host_vector<int> &hMod);

#endif //MODULE9_THRUST_HELPERFUNCTIONS_CUH
