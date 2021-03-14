#include "HelperFunctions.cuh"

__global__ void
kernelAdd(const int *a, const int *b, int *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = a[tid] + b[tid];
#ifdef  DEBUG
    printf("Thread ID %d: %d + %d = %d\n", tid, a[tid], b[tid], out[tid]);
#endif
}

__global__ void
kernelSubtract(const int *a, const int *b, int *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = a[tid] - b[tid];
#ifdef  DEBUG
    printf("Thread ID %d: %d - %d = %d\n", tid, a[tid], b[tid], out[tid]);
#endif
}

__global__ void
kernelMultiply(const int *a, const int *b, int *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = a[tid] * b[tid];
#ifdef  DEBUG
    printf("Thread ID %d: %d * %d = %d\n", tid, a[tid], b[tid], out[tid]);
#endif
}

__global__ void
kernelModulus(const int *a, const int *b, int *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] =  (a[tid] % b[tid]);
#ifdef  DEBUG
    printf("Thread ID %d: %d mod %d = %d\n", tid, a[tid], b[tid], out[tid]);
#endif
}

void hostAdd(const int *h_a, const int *h_b, const int numStreams,
             const int numBlocks, std::vector<float> &times)
{
    DEVICE_ALLOCATE(d_out, int, numStreams);
    DEVICE_ALLOCATE(d_a, int, numStreams);
    DEVICE_ALLOCATE(d_b, int, numStreams);
    ALLOCATE_PAGEABLE_MEMORY(h_out, int, numStreams);

    ACTIVATE_KERNEL(kernelAdd);

    times.push_back(delta);
    freeHostAlloc(h_out);
    freeDeviceAlloc(d_out, d_a, d_b);
}

void hostSub(const int *h_a, const int *h_b, const int numStreams,
             const int numBlocks, std::vector<float> &times)
{
    DEVICE_ALLOCATE(d_out, int, numStreams);
    DEVICE_ALLOCATE(d_a, int, numStreams);
    DEVICE_ALLOCATE(d_b, int, numStreams);
    ALLOCATE_PAGEABLE_MEMORY(h_out, int, numStreams);

    ACTIVATE_KERNEL(kernelSubtract);

    times.push_back(delta);
    freeHostAlloc(h_out);
    freeDeviceAlloc(d_out, d_a, d_b);
}

void hostMult(const int *h_a, const int *h_b, const int numStreams,
              const int numBlocks, std::vector<float> &times)
{
    DEVICE_ALLOCATE(d_out, int, numStreams);
    DEVICE_ALLOCATE(d_a, int, numStreams);
    DEVICE_ALLOCATE(d_b, int, numStreams);
    ALLOCATE_PAGEABLE_MEMORY(h_out, int, numStreams);

    ACTIVATE_KERNEL(kernelMultiply);

    times.push_back(delta);
    freeHostAlloc(h_out);
    freeDeviceAlloc(d_out, d_a, d_b);
}

void hostMod(const int *h_a, const int *h_b, const int numStreams,
             const int numBlocks, std::vector<float> &times)
{
    DEVICE_ALLOCATE(d_out, int, numStreams);
    DEVICE_ALLOCATE(d_a, int, numStreams);
    DEVICE_ALLOCATE(d_b, int, numStreams);
    ALLOCATE_PAGEABLE_MEMORY(h_out, int, numStreams);

    ACTIVATE_KERNEL(kernelModulus);

    times.push_back(delta);
    freeHostAlloc(h_out);
    freeDeviceAlloc(d_out, d_a, d_b);
}