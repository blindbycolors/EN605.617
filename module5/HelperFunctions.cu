#include "HelperFunctions.cuh"
#include <cstdio>
// Kernel Functions

__constant__ int constantArrA[NUM_ELEMENTS];
__constant__ int constantArrB[NUM_ELEMENTS];
__constant__ char constantInput[NUM_ELEMENTS];

__global__ void sharedAdd(const int* a, const int* b, int *out)
{
    __shared__ int sharedA[NUM_ELEMENTS];
    __shared__ int sharedB[NUM_ELEMENTS];
    __shared__ int sharedOut[NUM_ELEMENTS];
    sharedA[threadIdx.x] = a[threadIdx.x];
    sharedB[threadIdx.x] = b[threadIdx.x];
    __syncthreads();

    auto tid = blockIdx.x  * blockDim.x + threadIdx.x;
    sharedOut[threadIdx.x] = sharedA[threadIdx.x] + sharedB[threadIdx.x];
    __syncthreads();
    out[tid] = sharedOut[threadIdx.x];
#ifdef DEBUG
    printf("Dynamic Shared Memory Add: %d, f=%d + %d = %d\n",
           tid,
           a[tid],
           b[tid],
           out[tid]);
#endif // DEBUG

}

__global__ void sharedSub(const int* a, const int* b, int *out)
{
    __shared__ int sharedA[NUM_ELEMENTS];
    __shared__ int sharedB[NUM_ELEMENTS];
    __shared__ int sharedOut[NUM_ELEMENTS];
    sharedA[threadIdx.x] = a[threadIdx.x];
    sharedB[threadIdx.x] = b[threadIdx.x];
    __syncthreads();

    auto tid = blockIdx.x  * blockDim.x + threadIdx.x;
    sharedOut[threadIdx.x] = sharedA[threadIdx.x] - sharedB[threadIdx.x];
    __syncthreads();
    out[tid] = sharedOut[threadIdx.x];
#ifdef DEBUG
    printf("Dynamic Shared Memory Subtract: %d, f=%d - %d = %d\n",
           tid,
           a[tid],
           b[tid],
           out[tid]);
#endif // DEBUG
}

__global__ void sharedMult(const int* a, const int* b, int *out)
{
    __shared__ int sharedA[NUM_ELEMENTS];
    __shared__ int sharedB[NUM_ELEMENTS];
    __shared__ int sharedOut[NUM_ELEMENTS];
    sharedA[threadIdx.x] = a[threadIdx.x];
    sharedB[threadIdx.x] = b[threadIdx.x];
    __syncthreads();

    auto tid = blockIdx.x  * blockDim.x + threadIdx.x;
    sharedOut[threadIdx.x] = sharedA[threadIdx.x] * sharedB[threadIdx.x];
    __syncthreads();
    out[tid] = sharedOut[threadIdx.x];
#ifdef DEBUG
    printf("Dynamic Shared Memory Multiply: %d, f=%d * %d = %d\n",
           tid,
           a[tid],
           b[tid],
           out[tid]);
#endif // DEBUG
}

__global__ void sharedMod(const int* a, const int* b, int *out)
{
    __shared__ int sharedA[NUM_ELEMENTS];
    __shared__ int sharedB[NUM_ELEMENTS];
    __shared__ int sharedOut[NUM_ELEMENTS];
    sharedA[threadIdx.x] = a[threadIdx.x];
    sharedB[threadIdx.x] = b[threadIdx.x];
    __syncthreads();

    auto tid = blockIdx.x  * blockDim.x + threadIdx.x;
    sharedOut[threadIdx.x] = sharedA[threadIdx.x] % sharedB[threadIdx.x];
    __syncthreads();
    out[tid] = sharedOut[threadIdx.x];
#ifdef DEBUG
    printf("Dynamic Shared Memory Modulus: %d, f=%d mod %d = %d\n",
           tid,
           a[tid],
           b[tid],
           out[tid]);
#endif // DEBUG
}

__global__ void constantAdd(int * out)
{
    out[threadIdx.x] = constantArrA[threadIdx.x] + constantArrB[threadIdx.x];
#ifdef DEBUG
    printf("Constant Memory Add: %d, %d + %d = %d\n",
           threadIdx.x,
           constantArrA[threadIdx.x],
           constantArrB[threadIdx.x],
           out[threadIdx.x]);
#endif
}

__global__ void constantSub(int * out)
{
    out[threadIdx.x] = constantArrA[threadIdx.x] - constantArrB[threadIdx.x];
#ifdef DEBUG
    printf("Constant Memory Subtract: %d, %d - %d = %d\n",
           threadIdx.x,
           constantArrA[threadIdx.x],
           constantArrB[threadIdx.x],
           out[threadIdx.x]);
#endif
}

__global__ void constantMult(int * out)
{
    out[threadIdx.x] = constantArrA[threadIdx.x] * constantArrB[threadIdx.x];
#ifdef DEBUG
    printf("Constant Memory Multiply: %d, %d * %d = %d\n",
           threadIdx.x,
           constantArrA[threadIdx.x],
           constantArrB[threadIdx.x],
           out[threadIdx.x]);
#endif
}

__global__ void constantMod(int * out)
{
    out[threadIdx.x] = constantArrA[threadIdx.x] % constantArrB[threadIdx.x];
#ifdef DEBUG
    printf("Constant Memory Modulus: %d, %d mod %d = %d\n",
           threadIdx.x,
           constantArrA[threadIdx.x],
           constantArrB[threadIdx.x],
           out[threadIdx.x]);
#endif
}

__global__ void sharedCeaserEncrypt(const char *value,
                                    const int offset,
                                    char *d_out)
{
    __shared__ char sharedOut[NUM_ELEMENTS];
    __shared__ char sharedValue[NUM_ELEMENTS];

    sharedValue[threadIdx.x] = value[threadIdx.x];
    __syncthreads();

    sharedOut[threadIdx.x] = sharedValue[threadIdx.x] + offset % MAX_ASCII;
    __syncthreads();
#ifdef DEBUG
    printf("Encrypted: %d, %c\n",
           threadIdx.x,
           sharedOut[threadIdx.x]);
#endif // DEBUG
    d_out[threadIdx.x] = sharedOut[threadIdx.x];
}

__global__ void sharedCeaserDecrypt(const char *value,
                                    const int offset,
                                    char *d_out)
{
    __shared__ char sharedOut[NUM_ELEMENTS];
    __shared__ char sharedValue[NUM_ELEMENTS];

    sharedValue[threadIdx.x] = value[threadIdx.x];
    __syncthreads();

    auto tid = blockIdx.x  * blockDim.x + threadIdx.x;
    sharedOut[threadIdx.x] = sharedValue[threadIdx.x] - offset % MAX_ASCII;
    __syncthreads();
#ifdef DEBUG
    printf("Decrypted: %d, %c\n", threadIdx.x, sharedOut[threadIdx.x]);
#endif // DEBUG
    d_out[tid] = sharedOut[threadIdx.x];
}

__global__ void constantCaeserEncrypt(const int offset, char *out)
{
    out[threadIdx.x] = constantInput[threadIdx.x] + offset % MAX_ASCII;
#ifdef DEBUG
    printf("Encrypted: %d, %c\n", threadIdx.x, out[threadIdx.x]);
#endif
}

__global__ void constantCaeserDecrypt(const int offset, char *out)
{
    out[threadIdx.x] = constantInput[threadIdx.x] - offset % MAX_ASCII;
#ifdef DEBUG
    printf("Decrypted: %d, %c\n", threadIdx.x, out[threadIdx.x]);
#endif
}

// Wrapper Methods
void hostAdd(const int* h_a, const int* h_b)
{
    DEVICE_ALLOCATE(d_outShared, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_outConstant, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_a, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_b, int, NUM_ELEMENTS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));

    // Do Dynamic add
    printf("Shared Memory Add:\n");
    ACTIVATE_SHARED_OP_KERNEL(sharedAdd, d_a, d_b, d_outShared,
                              sharedKernelStart, sharedKernelStop,
                              dynamicDuration);

    // Do Constant Add
    printf("Constant Memory Add:\n");
    ACTIVATE_CONSTANT_OP_KERNEL(constantAdd, d_outConstant, h_a, h_b,
                                constantKernelStart, constantKernelStop,
                                constantDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant);
}

void hostSub(const int* h_a, const int* h_b)
{
    DEVICE_ALLOCATE(d_outShared, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_outConstant, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_a, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_b, int, NUM_ELEMENTS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));

    // Do Dynamic add
    printf("Shared Memory Subtract:\n");
    ACTIVATE_SHARED_OP_KERNEL(sharedSub, d_a, d_b, d_outShared,
                              sharedKernelStart, sharedKernelStop,
                              dynamicDuration);

    // Do Constant Add
    printf("Constant Memory Subtract:\n");
    ACTIVATE_CONSTANT_OP_KERNEL(constantSub, d_outConstant, h_a, h_b,
                                constantKernelStart, constantKernelStop,
                                constantDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant);
}

void hostMult(const int* h_a, const int* h_b)
{
    DEVICE_ALLOCATE(d_outShared, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_outConstant, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_a, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_b, int, NUM_ELEMENTS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));

    // Do Dynamic add
    printf("Shared Memory Multiply:\n");
    ACTIVATE_SHARED_OP_KERNEL(sharedMult, d_a, d_b, d_outShared,
                              sharedKernelStart, sharedKernelStop,
                              dynamicDuration);

    // Do Constant Add
    printf("Constant Memory Multiply:\n");
    ACTIVATE_CONSTANT_OP_KERNEL(constantMult, d_outConstant, h_a, h_b,
                                constantKernelStart, constantKernelStop,
                                constantDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant);
}

void hostMod(const int* h_a, const int* h_b)
{
    DEVICE_ALLOCATE(d_outShared, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_outConstant, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_a, int, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_b, int, NUM_ELEMENTS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));

    // Do Dynamic add
    printf("Shared Memory Modulus:\n");
    ACTIVATE_SHARED_OP_KERNEL(sharedMod, d_a, d_b, d_outShared,
                              sharedKernelStart, sharedKernelStop,
                              dynamicDuration);

    // Do Constant Add
    printf("Constant Memory Modulus:\n");
    ACTIVATE_CONSTANT_OP_KERNEL(constantMod, d_outConstant, h_a, h_b,
                                constantKernelStart, constantKernelStop,
                                constantDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant);
}

std::string hostSharedEncrypt(const char *h_input, int offset)
{
    DEVICE_ALLOCATE(d_input, char, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_out, char, NUM_ELEMENTS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, NUM_ELEMENTS);

    checkCuda(cudaMemcpy(d_input, h_input, sizeof(char) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));
    printf("Shared Memory Encrypt:\n");
    ACTIVATE_SHARED_CAESER_KERNEL(sharedCeaserEncrypt, d_input, d_out, offset);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * NUM_ELEMENTS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_input, d_out);
    freeHostAlloc(h_out);

    return retString;
}

std::string hostSharedDecrypt(const char *h_input, int offset)
{
    DEVICE_ALLOCATE(d_input, char, NUM_ELEMENTS);
    DEVICE_ALLOCATE(d_out, char, NUM_ELEMENTS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, NUM_ELEMENTS);

    checkCuda(cudaMemcpy(d_input, h_input, sizeof(char) * NUM_ELEMENTS,
                         cudaMemcpyHostToDevice));

    printf("Shared Memory Decrypt:\n");
    ACTIVATE_SHARED_CAESER_KERNEL(sharedCeaserDecrypt, d_input, d_out, offset);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * NUM_ELEMENTS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_input, d_out);
    freeHostAlloc(h_out);

    return retString;
}

std::string hostConstantEncrypt(const char *h_input, int offset)
{
    DEVICE_ALLOCATE(d_out, char, NUM_ELEMENTS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, NUM_ELEMENTS);

    printf("Constant Memory Encrypt:\n");
    ACTIVATE_CONSTANT_CAESER_KERNEL(constantCaeserEncrypt, d_out, h_input,
                                    offset);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * NUM_ELEMENTS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_out);
    freeHostAlloc(h_out);

    return retString;
}

std::string hostConstantDecrypt(const char *h_input, int offset)
{
    DEVICE_ALLOCATE(d_out, char, NUM_ELEMENTS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, NUM_ELEMENTS);

    printf("Constant Memory Encrypt:\n");
    ACTIVATE_CONSTANT_CAESER_KERNEL(constantCaeserDecrypt, d_out, h_input,
                                    offset);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * NUM_ELEMENTS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_out);
    freeHostAlloc(h_out);

    return retString;
}
