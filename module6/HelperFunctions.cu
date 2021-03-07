#include "HelperFunctions.cuh"
#include <cstdio>
#include <vector>

// Constant Arrays for Constant Memory Test
//-----------------------------------------------------------------------------
__constant__ int constantArrA[KERNEL_LOOPS];
__constant__ int constantArrB[KERNEL_LOOPS];
__constant__ char constantInput[KERNEL_LOOPS];

// Static shared memory kernel functions
//-----------------------------------------------------------------------------
__global__ void
sharedAdd(const int *a, const int *b, int *out, const int numElements)
{
    __shared__ int sharedA[KERNEL_LOOPS];
    __shared__ int sharedB[KERNEL_LOOPS];
    __shared__ int sharedOut[KERNEL_LOOPS];
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        sharedA[tid] = a[tid];
        sharedB[tid] = b[tid];
        __syncthreads();

        sharedOut[tid] = sharedA[tid] + sharedB[tid];
        __syncthreads();

        out[tid] = sharedOut[tid];
#ifdef DEBUG
        printf("Shared Memory Add: %d, f=%d + %d = %d\n", tid, sharedA[tid],
               sharedB[tid], out[tid]);
#endif // DEBUG
    }
}

__global__ void
sharedSub(const int *a, const int *b, int *out, const int numElements)
{
    __shared__ int sharedA[KERNEL_LOOPS];
    __shared__ int sharedB[KERNEL_LOOPS];
    __shared__ int sharedOut[KERNEL_LOOPS];
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        sharedA[tid] = a[tid];
        sharedB[tid] = b[tid];
        __syncthreads();

        sharedOut[tid] = sharedA[tid] - sharedB[tid];
        __syncthreads();
        out[tid] = sharedOut[tid];
#ifdef DEBUG
        printf("Shared Memory Subtract: %d, f=%d - %d = %d\n", tid, sharedA[tid],
               sharedB[tid], out[tid]);
#endif // DEBUG
    }
}

__global__ void
sharedMult(const int *a, const int *b, int *out, const int numElements)
{
    __shared__ int sharedA[KERNEL_LOOPS];
    __shared__ int sharedB[KERNEL_LOOPS];
    __shared__ int sharedOut[KERNEL_LOOPS];
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements)
    {
        sharedA[tid] = a[tid];
        sharedB[tid] = b[tid];
        __syncthreads();

        sharedOut[tid] = sharedA[tid] * sharedB[tid];
        __syncthreads();

        out[tid] = sharedOut[tid];
#ifdef DEBUG
        printf("Shared Memory Multiply: %d, f=%d * %d = %d\n", tid, sharedA[tid],
               sharedB[tid], out[tid]);
#endif // DEBUG
    }
}

__global__ void
sharedMod(const int *a, const int *b, int *out, const int numElements)
{
    __shared__ int sharedA[KERNEL_LOOPS];
    __shared__ int sharedB[KERNEL_LOOPS];
    __shared__ int sharedOut[KERNEL_LOOPS];
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements)
    {
        sharedA[tid] = a[tid];
        sharedB[tid] = b[tid];
        __syncthreads();

        sharedOut[tid] = sharedA[tid] % sharedB[tid];
        __syncthreads();
        out[tid] = sharedOut[tid];
#ifdef DEBUG
        printf("Shared Memory Modulus: %d, f=%d mod %d = %d\n", tid, sharedA[tid],
               sharedB[tid], out[tid]);
#endif // DEBUG
    }
}


__global__ void
sharedCeaserEncrypt(const char *value, int offset, char *d_out,
                    const int numElements)
{
    __shared__ char sharedOut[KERNEL_LOOPS];
    __shared__ char sharedValue[KERNEL_LOOPS];
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < numElements)
    {
        sharedValue[tid] = value[tid];
        __syncthreads();

        sharedOut[tid] = sharedValue[tid] + offset % MAX_ASCII;
        __syncthreads();
        d_out[tid] = sharedOut[tid];
#ifdef DEBUG
        printf("%d : Original: %c, Encrypted: %c\n", tid, sharedValue[tid],
               sharedOut[tid]);
#endif // DEBUG
    }
}

__global__ void
sharedCeaserDecrypt(const char *value, int offset, char *d_out,
                    const int numElements)
{
    __shared__ char sharedOut[KERNEL_LOOPS];
    __shared__ char sharedValue[KERNEL_LOOPS];
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numElements)
    {
        sharedValue[tid] = value[tid];
        __syncthreads();

        sharedOut[tid] = sharedValue[tid] - offset % MAX_ASCII;
        __syncthreads();
        d_out[tid] = sharedOut[tid];
#ifdef DEBUG
        printf("%d : Original: %c, Decrypted: %c\n", tid, sharedValue[tid],
               sharedOut[tid]);
#endif // DEBUG
    }
}

// Constant memory kernel functions
//-----------------------------------------------------------------------------
__global__ void constantAdd(int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        out[tid] = constantArrA[tid] + constantArrB[tid];
#ifdef DEBUG
        printf("Constant Memory Add: %d, %d + %d = %d\n", tid,
               constantArrA[tid], constantArrB[tid], out[tid]);
#endif
    }
}

__global__ void constantSub(int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        out[tid] =
                constantArrA[tid] - constantArrB[tid];
#ifdef DEBUG
        printf("Constant Memory Subtract: %d, %d - %d = %d\n", tid,
               constantArrA[tid], constantArrB[tid], out[tid]);
#endif
    }
}

__global__ void constantMult(int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        out[tid] =
                constantArrA[tid] * constantArrB[tid];
#ifdef DEBUG
        printf("Constant Memory Multiply: %d, %d * %d = %d\n", tid,
               constantArrA[tid], constantArrB[tid], out[tid]);
#endif
    }
}

__global__ void constantMod(int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        out[tid] = constantArrA[tid] % constantArrB[tid];
#ifdef DEBUG
        printf("Constant Memory Modulus: %d, %d mod %d = %d\n", tid,
               constantArrA[tid], constantArrB[tid],
               out[tid]);
#endif
    }
}

__global__ void
constantCaeserEncrypt(int offset, char *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        out[tid] = constantInput[tid] + offset % MAX_ASCII;
#ifdef DEBUG
        printf("Encrypted: %d, %c\n", tid, out[tid]);
#endif
    }
}

__global__ void
constantCaeserDecrypt(int offset, char *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        out[tid] = constantInput[tid] - offset % MAX_ASCII;
#ifdef DEBUG
        printf("Decrypted: %d, %c\n", tid, out[tid]);
#endif
    }
}

// Register memory kernel functions
//-----------------------------------------------------------------------------
__global__ void
registerAdd(const int *a, const int *b, int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        int tempA = a[tid];
        int tempB = b[tid];
        int tempOut = tempA + tempB;
        out[tid] = tempOut;
#ifdef DEBUG
        printf("Register Add %d : %d + %d = %d\n", tempA, tempB, tempOut);
#endif
    }
}

__global__ void
registerSub(const int *a, const int *b, int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        int tempA = a[tid];
        int tempB = b[tid];
        int tempOut = tempA - tempB;
        out[tid] = tempOut;
#ifdef DEBUG
        printf("Register Subtract %d : %d - %d = %d\n", tempA, tempB, tempOut);
#endif
    }
}

__global__ void
registerMult(const int *a, const int *b, int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        int tempA = a[tid];
        int tempB = b[tid];
        int tempOut = tempA * tempB;
        out[tid] = tempOut;
#ifdef DEBUG
        printf("Register Multiply %d : %d * %d = %d\n", tempA, tempB, tempOut);
#endif
    }
}

__global__ void
registerMod(const int *a, const int *b, int *out, const int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        int tempA = a[tid];
        int tempB = b[tid];
        int tempOut = tempA % tempB;
        out[tid] = tempOut;
#ifdef DEBUG
        printf("Register Modulus %d : %d % %d = %d\n", tempA, tempB, tempOut);
#endif
    }
}


__global__ void
registerCaeserEncrypt(const char *value, int offset, char *d_out,
                      int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        char temp = value[tid];
        temp = temp + offset % MAX_ASCII;
        d_out[tid] = temp;
#ifdef DEBUG
        printf("Encrypted: %d, %c\n", tid, temp);
#endif
    }
}

__global__ void
registerCaeserDecrypt(const char *value, int offset, char *d_out,
                      int numElements)
{
    auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < numElements)
    {
        char temp = value[tid];
        temp = temp - offset % MAX_ASCII;
        d_out[tid] = temp;
#ifdef DEBUG
        printf("Decrypted: %d, %c\n", tid, temp);
#endif
    }
}


// Wrapper Methods
//-----------------------------------------------------------------------------
void hostAdd(const int *h_a, const int *h_b, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_outShared, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outConstant, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outRegister, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_a, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_b, int, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));

    ACTIVATE_SHARED_OP_KERNEL(sharedAdd, sharedKernelStart, sharedKernelStop,
                              dynamicDuration);
    ACTIVATE_CONSTANT_OP_KERNEL(constantAdd, constantKernelStart,
                                constantKernelStop, constantDuration);
    ACTIVATE_REGISTER_OP_KERNEL(registerAdd, regKernelStart, regKernelStop,
                                regDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant, d_outRegister);
}

void hostSub(const int *h_a, const int *h_b, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_outShared, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outConstant, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outRegister, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_a, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_b, int, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));

    ACTIVATE_SHARED_OP_KERNEL(sharedSub, sharedKernelStart, sharedKernelStop,
                              dynamicDuration);
    ACTIVATE_CONSTANT_OP_KERNEL(constantSub, constantKernelStart,
                                constantKernelStop, constantDuration);
    ACTIVATE_REGISTER_OP_KERNEL(registerSub, regKernelStart, regKernelStop,
                                regDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant, d_outRegister);
}

void hostMult(const int *h_a, const int *h_b, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_outShared, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outConstant, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outRegister, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_a, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_b, int, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));

    ACTIVATE_SHARED_OP_KERNEL(sharedMult, sharedKernelStart, sharedKernelStop,
                              dynamicDuration);
    ACTIVATE_CONSTANT_OP_KERNEL(constantMult, constantKernelStart,
                                constantKernelStop, constantDuration);
    ACTIVATE_REGISTER_OP_KERNEL(registerMult, regKernelStart, regKernelStop,
                                regDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant, d_outRegister);
}

void hostMod(const int *h_a, const int *h_b, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_outShared, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outConstant, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_outRegister, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_a, int, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_b, int, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));

    ACTIVATE_SHARED_OP_KERNEL(sharedMod, sharedKernelStart, sharedKernelStop,
                              dynamicDuration);
    ACTIVATE_CONSTANT_OP_KERNEL(constantMod,  constantKernelStart,
                                constantKernelStop, constantDuration);
    ACTIVATE_REGISTER_OP_KERNEL(registerMod, regKernelStart, regKernelStop,
                                regDuration);

    freeDeviceAlloc(d_a, d_b, d_outShared, d_outConstant, d_outRegister);
}

std::string hostSharedEncrypt(const char *h_input, const int offset,
                              std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_input, char, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_out, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_input, h_input, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));
    ACTIVATE_SHARED_CAESER_KERNEL(sharedCeaserEncrypt);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_input, d_out);
    freeHostAlloc(h_out);

    return retString;
}

std::string hostSharedDecrypt(const char *h_input, const int offset,
                              std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_input, char, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_out, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_input, h_input, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));

    ACTIVATE_SHARED_CAESER_KERNEL(sharedCeaserDecrypt);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_input, d_out);
    freeHostAlloc(h_out);

    return retString;
}

std::string
hostConstantEncrypt(const char *h_input, int offset, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_out, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, KERNEL_LOOPS);

    ACTIVATE_CONSTANT_CAESER_KERNEL(constantCaeserEncrypt);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_out);
    freeHostAlloc(h_out);

    return retString;
}

std::string
hostConstantDecrypt(const char *h_input, int offset, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_out, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, KERNEL_LOOPS);

    ACTIVATE_CONSTANT_CAESER_KERNEL(constantCaeserDecrypt);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_out);
    freeHostAlloc(h_out);

    return retString;
}


std::string
hostRegisterEncrypt(const char *h_input, int offset, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_input, char, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_out, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_input, h_input, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));

    ACTIVATE_REGISTER_CAESER_KERENEL(registerCaeserEncrypt);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_out);
    freeHostAlloc(h_out);

    return retString;
}

std::string
hostRegisterDecrypt(const char *h_input, int offset, std::vector<float> &time)
{
    CALC_NUM_BLOCKS();
    DEVICE_ALLOCATE(d_input, char, KERNEL_LOOPS);
    DEVICE_ALLOCATE(d_out, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(h_out, char, KERNEL_LOOPS);

    checkCuda(cudaMemcpy(d_input, h_input, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyHostToDevice));

    ACTIVATE_REGISTER_CAESER_KERENEL(registerCaeserDecrypt);

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(char) * KERNEL_LOOPS,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_out);

    freeDeviceAlloc(d_out);
    freeHostAlloc(h_out);

    return retString;
}

