#include <string>
#include <cassert>
#include <vector>

#ifndef MODULE5_HELPERFUNCTIONS_CUH
#define MODULE5_HELPERFUNCTIONS_CUH

#define MIN 1
#define MAX 3
#define KERNEL_LOOPS 1024
#define MIN_PRINTABLE 32
#define MAX_PRINTABLE 126
#define MAX_ASCII 128

#define CALC_NUM_BLOCKS()                               \
    auto numElements = KERNEL_LOOPS;                    \
    auto numThreads = KERNEL_LOOPS;                     \
    auto numBlocks = numElements / numThreads;          \
    // End CALC_NUM_BLOCKS

#define HOST_ALLOCATE_PAGEABLE(a, type, total)          \
    type* a = (type *) malloc(sizeof(type) * total);    \
    // End HOST_ALLOCATE_PAGEABLE

#define HOST_ALLOCATE_PINNED(a, type, total)                                   \
    type * a;                                                                  \
    checkCuda(cudaMallocHost((void**) &a, sizeof(type) * total));              \
    // End HOST_ALLOCATE_PINNED

#define ACTIVATE_SHARED_OP_KERNEL(MATH_OP, KERNEL_START, KERNEL_STOP, DELTA)   \
    cudaEvent_t KERNEL_START, KERNEL_STOP;                                     \
    auto DELTA = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&KERNEL_START));                                 \
    checkCuda(cudaEventCreate(&KERNEL_STOP));                                  \
    checkCuda(cudaEventRecord(KERNEL_START,0));                                \
    MATH_OP<<<numBlocks, numThreads>>>(d_a, d_b, d_outShared, numElements);    \
    checkCuda(cudaEventRecord(KERNEL_STOP,0));                                 \
    checkCuda(cudaEventSynchronize(KERNEL_STOP));                              \
    checkCuda(cudaEventElapsedTime(&DELTA, KERNEL_START, KERNEL_STOP));        \
    checkCuda(cudaEventDestroy(KERNEL_START));                                 \
    checkCuda(cudaEventDestroy(KERNEL_STOP));                                  \
    time.push_back(DELTA);                                                     \
    // End ACTIVATE_SHARED_OP_KERNEL macro

#define ACTIVATE_CONSTANT_OP_KERNEL(MATH_OP, KERNEL_START, KERNEL_STOP, DELTA) \
    cudaEvent_t KERNEL_START, KERNEL_STOP;                                     \
    cudaMemcpyToSymbol(constantArrA, h_a, sizeof(int) * KERNEL_LOOPS);         \
    cudaMemcpyToSymbol(constantArrB, h_b, sizeof(int) * KERNEL_LOOPS);         \
    auto DELTA = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&KERNEL_START));                                 \
    checkCuda(cudaEventCreate(&KERNEL_STOP));                                  \
    checkCuda(cudaEventRecord(KERNEL_START,0));                                \
    MATH_OP<<<numBlocks, numThreads>>>(d_outConstant, numElements);            \
    checkCuda(cudaEventRecord(KERNEL_STOP,0));                                 \
    checkCuda(cudaEventSynchronize(KERNEL_STOP));                              \
    checkCuda(cudaEventElapsedTime(&DELTA, KERNEL_START, KERNEL_STOP));        \
    checkCuda(cudaEventDestroy(KERNEL_START));                                 \
    checkCuda(cudaEventDestroy(KERNEL_STOP));                                  \
    time.push_back(DELTA);                                                     \
    // End ACTIVATE_CONSTANT_OP_KERNEL

#define ACTIVATE_REGISTER_OP_KERNEL(MATH_OP, KERNEL_START, KERNEL_STOP,      \
    DELTA)                                                                   \
        cudaEvent_t KERNEL_START, KERNEL_STOP;                               \
        auto DELTA = 0.0F;                                                   \
        checkCuda(cudaEventCreate(&KERNEL_START));                           \
        checkCuda(cudaEventCreate(&KERNEL_STOP));                            \
        checkCuda(cudaEventRecord(KERNEL_START,0));                          \
        MATH_OP<<<numBlocks, numThreads>>>(d_a, d_b, d_outRegister,          \
            numElements);                                                    \
        checkCuda(cudaEventRecord(KERNEL_STOP,0));                           \
        checkCuda(cudaEventSynchronize(KERNEL_STOP));                        \
        checkCuda(cudaEventElapsedTime(&DELTA, KERNEL_START, KERNEL_STOP));  \
        checkCuda(cudaEventDestroy(KERNEL_START));                           \
        checkCuda(cudaEventDestroy(KERNEL_STOP));                            \
        time.push_back(DELTA);                                               \
    // End ACTIVATE_REGISTER_OP_KERNEL

#define ACTIVATE_SHARED_CAESER_KERNEL(OPERATION)                               \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart, 0));                                \
    OPERATION<<<numBlocks, numThreads>>>(d_input, offset, d_out, numElements); \
    checkCuda(cudaEventRecord(kernelStop, 0));                                 \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));                                   \
    time.push_back(delta);                                                     \
    // End ACTIVATE_SHARED_CAESER_KERNEL macro

#define ACTIVATE_CONSTANT_CAESER_KERNEL(OPERATION)                             \
    cudaMemcpyToSymbol(constantInput, h_input, sizeof(char) * KERNEL_LOOPS);   \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart, 0));                                \
    OPERATION<<<numBlocks, numThreads>>>(offset, d_out, numElements);          \
    checkCuda(cudaEventRecord(kernelStop, 0));                                 \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));                                   \
    time.push_back(delta);                                                     \
    // End ACTIVATE_SHARED_CAESER_KERNEL macro

#define ACTIVATE_REGISTER_CAESER_KERENEL(OPERATION)                            \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0f;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart, 0));                                \
    OPERATION<<<numBlocks, numThreads>>>(d_input, offset, d_out, numElements); \
    checkCuda(cudaEventRecord(kernelStop, 0));                                 \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));                                   \
    time.push_back(delta);                                                     \

#define DEVICE_ALLOCATE(a, type, totalThreads)                                 \
    type *a;                                                                   \
    checkCuda(cudaMalloc((void**) &a, sizeof(type) * totalThreads));           \
    // End DEVICE_ALLOCATE macro

// Convenience function for checking CUDA runtime API results
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Shared Memory Kernel functions
__global__ void
sharedAdd(const int *a, const int *b, int *out, int numElements);

__global__ void
sharedSub(const int *a, const int *b, int *out, int numElements);

__global__ void
sharedMult(const int *a, const int *b, int *out, int numElements);

__global__ void
sharedMod(const int *a, const int *b, int *out, int numElements);

__global__ void sharedCeaserEncrypt(const char *value, int offset, char *d_out,
                                    int numElements);

__global__ void sharedCeaserDecrypt(const char *value, int offset, char *d_out,
                                    int numElements);

// Constant Memory Kernel functions
__global__ void constantAdd(int *out, int numElements);

__global__ void constantSub(int *out, int numElements);

__global__ void constantMult(int *out, int numElements);

__global__ void constantMod(int *out, int numElements);

__global__ void constantCaeserEncrypt(int offset, char *out, int numElements);

__global__ void constantCaeserDecrypt(int offset, char *out, int numElements);


// Register memory Kernel Functions
__global__ void
registerAdd(const int *a, const int *b, int *out, int numElements);

__global__ void registerSub(const int *a, const int *b, int *out, int
numElements);

__global__ void registerMult(const int *a, const int *b, int *out, int
numElements);

__global__ void registerMod(const int *a, const int *b, int *out, int
numElements);

__global__ void registerCaeserEncrypt(const char *value, int offset, char *d_out,
                                      int numElements);

__global__ void registerCaeserDecrypt(const char *value, int offset, char *d_out,
                                      int numElements);

// Utility Methods
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

void hostAdd(const int *h_a, const int *h_b, std::vector<float> &time);

void hostSub(const int *h_a, const int *h_b, std::vector<float> &time);

void hostMult(const int *h_a, const int *h_b, std::vector<float> &time);

void hostMod(const int *h_a, const int *h_B, std::vector<float> &time);

std::string hostSharedEncrypt(const char *h_input, int offset,
                              std::vector<float> &time);

std::string hostSharedDecrypt(const char *h_input, int offset,
                              std::vector<float> &time);

std::string
hostConstantEncrypt(const char *h_input, int offset, std::vector<float> &time);

std::string
hostConstantDecrypt(const char *h_input, int offset, std::vector<float> &time);

std::string
hostRegisterEncrypt(const char *h_input, int offset, std::vector<float> &time);

std::string
hostRegisterDecrypt(const char *h_input, int offset, std::vector<float> &time);

#endif //MODULE5_HELPERFUNCTIONS_CUH
