#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include "HelperFunctions.cuh"

void runPaged(const int N)
{
    // Allocate paged memroy
    ALLOCATE_PAGEABLE_MEMORY(hostA, int, N);
    ALLOCATE_PAGEABLE_MEMORY(hostB, int, N);

    // ALlocate memory for computed data
    thrust::host_vector<int> hAdd (N);
    thrust::host_vector<int> hSub (N);
    thrust::host_vector<int> hMul (N);
    thrust::host_vector<int> hMod (N);

    // Generate random values
    thrust::generate(hostA, hostA + N, randEngine);
    thrust::generate(hostB, hostB + N, randEngine);

    runOperations(hostA, hostB, N, hAdd, hSub, hMul, hMod);

#ifdef DEBUG
    printAll(hostA, hostB, N, hAdd, hSub, hMul, hMod);
#endif

    freeHostAlloc(hostA, hostB);
}

void runPinned(const int N)
{
    // Allocate paged memroy
    ALLOCATE_PINNED_MEMORY(hostA, int, N);
    ALLOCATE_PINNED_MEMORY(hostB, int, N);

    // ALlocate memory for computed data
    thrust::host_vector<int> hAdd (N);
    thrust::host_vector<int> hSub (N);
    thrust::host_vector<int> hMul (N);
    thrust::host_vector<int> hMod (N);

    // Generate random values
    thrust::generate(hostA, hostA + N, randEngine);
    thrust::generate(hostB, hostB + N, randEngine);

    runOperations(hostA, hostB, N, hAdd, hSub, hMul, hMod);

#ifdef DEBUG
    printAll(hostA, hostB, N, hAdd, hSub, hMul, hMod);
#endif

    freePinned(hostA, hostB);
}

int main(int argc, char **argv)
{
    // TODO get n from cmd input
    auto numSize = 1 << 20;

    if (argc == 2)
    {
        numSize = std::atoi(argv[1]);
    }

    printf("Paged Memory Runtime:\n");
    runPaged(numSize);
    printf("\n\n=========================================================\n\n");
    printf("Pinned Memory Runtime:\n");
    runPinned(numSize);
}