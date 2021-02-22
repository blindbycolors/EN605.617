//
// Created by nou on 2/21/21.
//

#include <cstdio>
#include "operations.cuh"

#define ACTIVATE_OP_KERNEL(opFunc, numBlocks, blockSize, d_a, d_b, d_out)      \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart,0));                                 \
    opFunc<<<numBlocks, blockSize>>>(d_a, d_b, d_out);                         \
    checkCuda(cudaEventRecord(kernelStop,0));                                  \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));

#define ACTIVATE_CAESER_KERNEL(opFunction, numBlocks, blockSize, totalThreads, \
                               d_a, d_out, offset)                             \
    cudaEvent_t kernelStart, kernelStop;                                       \
    auto delta = 0.0F;                                                         \
    checkCuda(cudaEventCreate(&kernelStart));                                  \
    checkCuda(cudaEventCreate(&kernelStop));                                   \
    checkCuda(cudaEventRecord(kernelStart, 0));                                \
    opFunction<<<numBlocks, blockSize>>>(d_a, totalThreads, offset, d_out);    \
    checkCuda(cudaEventRecord(kernelStop, 0));                                 \
    checkCuda(cudaEventSynchronize(kernelStop));                               \
    checkCuda(cudaEventElapsedTime(&delta, kernelStart, kernelStop));          \
    checkCuda(cudaEventDestroy(kernelStart));                                  \
    checkCuda(cudaEventDestroy(kernelStop));

#define DEVICE_ALLOCATE(a, type, totalThreads)                                 \
    type *a;                                                                   \
    checkCuda(cudaMalloc((void**) &a, sizeof(type) * totalThreads));

__global__ void kernel_addition(const int *a, const int *b, float *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = a[tid] + b[tid];
}

__global__ void kernel_subtraction(const int *a, const int *b, float *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = a[tid] - b[tid];
}

__global__ void kernel_multiplication(const int *a, const int *b, float *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = a[tid] * b[tid];
}

__global__ void kernel_modulus(const int *a, const int *b, float *out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = (float) (a[tid] % b[tid]);
}

__global__ void kernel_caeser_encrypt(const char *value, const int n,
                                      const int offset, char *d_out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        auto encryptedVal = value[tid] + offset % 128;
        d_out[tid] = encryptedVal;
    }
}

__global__ void kernel_caeser_decrypt(const char *value, const int n,
                                      const int offset, char *d_out)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        auto decryptedVal = value[tid] - offset % 128 ;
        d_out[tid] = decryptedVal;
    }
}


float hostAdd(const int numBlocks, const int blockSize,
              const int totalThreads, int *h_a, int *h_b)
{
    // Device Array
    DEVICE_ALLOCATE(d_out, float, totalThreads);
    DEVICE_ALLOCATE(d_a, int, totalThreads);
    DEVICE_ALLOCATE(d_b, int, totalThreads);

    // Copy from host to device
    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));

    // Run the kernel function
    ACTIVATE_OP_KERNEL(kernel_addition, numBlocks, blockSize, d_a, d_b,
                       d_out);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return delta;
}

float hostSubtract(const int numBlocks, const int blockSize,
                   const int totalThreads, int *h_a, int *h_b)
{
    // Device Array
    DEVICE_ALLOCATE(d_out, float, totalThreads);
    DEVICE_ALLOCATE(d_a, int, totalThreads);
    DEVICE_ALLOCATE(d_b, int, totalThreads);

    // Copy from host to device
    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));

    // Run the kernel function
    ACTIVATE_OP_KERNEL(kernel_subtraction, numBlocks, blockSize, d_a, d_b,
                       d_out);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return delta;
}


float hostMultiply(const int numBlocks, const int blockSize,
                   const int totalThreads, int *h_a, int *h_b)
{
    // Device Array
    DEVICE_ALLOCATE(d_out, float, totalThreads);
    DEVICE_ALLOCATE(d_a, int, totalThreads);
    DEVICE_ALLOCATE(d_b, int, totalThreads);

    // Copy from host to device
    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));

    // Run the kernel function
    ACTIVATE_OP_KERNEL(kernel_multiplication, numBlocks, blockSize, d_a, d_b,
                       d_out);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return delta;
}

float hostMod(const int numBlocks, const int blockSize,
                     const int totalThreads, int *h_a, int *h_b)
{
    // Device Array
    DEVICE_ALLOCATE(d_out, float, totalThreads);
    DEVICE_ALLOCATE(d_a, int, totalThreads);
    DEVICE_ALLOCATE(d_b, int, totalThreads);

    // Copy from host to device
    checkCuda(cudaMemcpy(d_a, h_a, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, sizeof(int) * totalThreads,
                         cudaMemcpyHostToDevice));

    // Run the kernel function
    ACTIVATE_OP_KERNEL(kernel_modulus, numBlocks, blockSize, d_a, d_b, d_out);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return delta;
}

std::string
hostEncrypt(const int numBlocks, const int blockSize,
            const char *h_aInput, int strLength, int offset)
{
    DEVICE_ALLOCATE(d_a, char, strLength);
    DEVICE_ALLOCATE(d_out, char, strLength);

    auto h_outEncrypt = (char *) malloc(sizeof(char) * strLength);
    checkCuda(cudaMemcpy(d_a, h_aInput, sizeof(char) * strLength,
                         cudaMemcpyHostToDevice));

    ACTIVATE_CAESER_KERNEL(kernel_caeser_encrypt, numBlocks, blockSize,
                           strLength, d_a, d_out, offset);

    printf("Encryption Duration: %f ms\n", delta);

    checkCuda(cudaMemcpy(h_outEncrypt, d_out, sizeof(char) * strLength,
                         cudaMemcpyDeviceToHost));
    std::string retString(h_outEncrypt);

    cudaFree(d_a);
    cudaFree(d_out);
    free(h_outEncrypt);

    return retString;
}

std::string
hostDecrypt(const int numBlocks, const int blockSize,
            const char *h_aInput, int strLength, int offset)
{
    DEVICE_ALLOCATE(d_a, char, strLength);
    DEVICE_ALLOCATE(d_out, char, strLength);

    auto h_outEncrypt = (char *) malloc(sizeof(char) * strLength);
    checkCuda(cudaMemcpy(d_a, h_aInput, sizeof(char) * strLength,
                         cudaMemcpyHostToDevice));
    ACTIVATE_CAESER_KERNEL(kernel_caeser_decrypt, numBlocks, blockSize,
                           strLength, d_a, d_out, offset);

    printf("Decryption Duration: %f ms\n", delta);

    checkCuda(cudaMemcpy(h_outEncrypt, d_out, sizeof(char) * strLength,
                         cudaMemcpyDeviceToHost));

    std::string retString(h_outEncrypt);

    cudaFree(d_a);
    cudaFree(d_out);
    free(h_outEncrypt);

    return retString;
}
