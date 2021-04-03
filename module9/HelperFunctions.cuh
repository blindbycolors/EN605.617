#ifndef MODULE9_NPP_NVGRAPH_HELPERFUNCTIONS_CUH
#define MODULE9_NPP_NVGRAPH_HELPERFUNCTIONS_CUH

// Helper macros
#include <nvgraph.h>
#include <iostream>
#include <random>
#include <assert.h>

#define NUM_VERTEX 6
#define NUM_EDGES 10
#define VERTEX_NUM_SETS 2
#define EDGE_NUM_SETS 1

#define ALLOCATE_PAGEABLE_MEMORY(a, type, total)        \
    type* a = (type *) malloc(sizeof(type) * total);    \
    // End ALLOCATE_PAGEABLE_MEMORY

#define ALLOCATE_PINNED_MEMORY(a, type, total)                     \
    type * a;                                                      \
    checkCuda(cudaMallocHost((void**) &a, sizeof(type) * total));  \
    // End ALLOCATE_PINNED_MEMORY

template<typename T>
void freeMalloc(T var)
{
    if (var)
    {
        free(var);
    }
}

template<typename T, typename ... Types>
void freeMalloc(T var1, Types ... vars)
{
    if(var1)
    {
        free(var1);
    }
    freeMalloc(vars...);
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

template<typename T>
void nppiFreeVars(T var)
{
    if(var)

    {
        nppiFree(var);
    }
}

template <typename T, typename ... Types>
void nppiFreeVars(T var, Types ... vars)
{
    if(var)
    {
        nppiFree(var);
    }

    nppiFreeVars(vars...);
}

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

inline void checkNvGraph(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        fprintf(stderr, "nvGraph Runtime Error : %s\n",nvgraphStatusGetString
        (status));
    }
}

void printResults(float * widestPathResults);

void buildGraph(float * h_weights,
                int * h_destinationOffsets,
                int * h_sourceIndices);

void runNvGraphWidestPath(int * h_destinationOffsets,
                          int * h_sourceIndices,
                          float * h_weights,
                          float * h_widestPath1,
                          float * h_widestPath2,
                          void** vertexDim);
#endif //MODULE9_NPP_NVGRAPH_HELPERFUNCTIONS_CUH
