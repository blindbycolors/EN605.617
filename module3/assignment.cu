#include <chrono>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#define MIN 1
#define MAX 3

enum Operations { ADDITION, SUBTRACTION, MULTIPLICATION, MODULUS };

__global__ void
do_addition(int *a, int *b, float *out, int n)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        out[tid] = a[tid] + b[tid];
    }
}

__global__ void
do_subtraction(int *a, int *b, float *out, int n)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        out[tid] = a[tid] - b[tid];
    }
}

__global__ void
do_multiplication(int *a, int *b, float *out, int n)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        out[tid] = a[tid] * b[tid];
    }
}

__global__ void
do_modulus(int *a, int *b, float *out, int n)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        out[tid] =  (float) (a[tid] % b[tid]);
    }
}

void
printDeviceInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (unsigned int i = 0; i < nDevices; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device Name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

void
printResults(const float *out, const int totalThreads,
             const std::string results, const unsigned int time)
{
    printf("%s took %u nanoSeconds\n",
           results.c_str(),
           time);
    for (unsigned int i = 0; i < totalThreads; ++i)
    {
        // add new line every 50 element
        if ((i + 1) % 25 == 0)
        {
            printf("%.2f\n", out[i]);
        }
        else
        {
            printf("%.2f\t", out[i]);
        }
    }
    printf("\n----------------------------------------------------\n");
}

void
doOperations(int totalThreads, int numBlocks, int blockSize)
{
    // allocate memory for arrays
    int *a = (int *) malloc(sizeof(int) * totalThreads);
    int *b = (int *) malloc(sizeof(int) * totalThreads);
    float *addOut = (float *) malloc(sizeof(float) * totalThreads);
    float *subOut = (float *) malloc(sizeof(float) * totalThreads);
    float *multOut = (float *) malloc(sizeof(float) * totalThreads);
    float *modOut = (float *) malloc(sizeof(float) * totalThreads);

    // Initialize arrays
    for (unsigned int i = 0; i < totalThreads; ++i)
    {
        a[i] = i;
        b[i] = rand() % MAX + MIN;
    }

    // Allocate device memory for arrays
    int *d_a, *d_b;
    float *d_addOut, *d_subOut, *d_multOut, *d_modOut;
    cudaMalloc((void **) &d_a, sizeof(int) * totalThreads);
    cudaMalloc((void **) &d_b, sizeof(int) * totalThreads);
    cudaMalloc((void **) &d_addOut, sizeof(float) * totalThreads);
    cudaMalloc((void **) &d_subOut, sizeof(float) * totalThreads);
    cudaMalloc((void **) &d_multOut, sizeof(float) * totalThreads);
    cudaMalloc((void **) &d_modOut, sizeof(float) * totalThreads);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(int) * totalThreads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * totalThreads, cudaMemcpyHostToDevice);


    // perform the operations
    auto start = std::chrono::high_resolution_clock::now();
    do_addition<<<numBlocks, blockSize>>>(d_a, d_b, d_addOut, totalThreads);
    auto addStop = std::chrono::high_resolution_clock::now();
    do_subtraction<<<numBlocks, blockSize>>>(d_a, d_b, d_subOut, totalThreads);
    auto subStop = std::chrono::high_resolution_clock::now();
    do_multiplication<<<numBlocks,
    blockSize>>>(d_a, d_b, d_multOut, totalThreads);
    auto multStop = std::chrono::high_resolution_clock::now();
    do_modulus<<<numBlocks, blockSize>>>(d_a, d_b, d_modOut, totalThreads);
    auto modStop = std::chrono::high_resolution_clock::now();

    // Transfer data from device to host
    cudaMemcpy(addOut,
               d_addOut,
               sizeof(float) * totalThreads,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(subOut, d_subOut, sizeof(float) * totalThreads,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(multOut, d_multOut, sizeof(float) * totalThreads,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(modOut, d_modOut, sizeof(float) * totalThreads,
               cudaMemcpyDeviceToHost);

    // clean up after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_addOut);
    cudaFree(d_subOut);
    cudaFree(d_multOut);
    cudaFree(d_modOut);

    printResults(addOut, totalThreads,
                 std::string("Addition Results"),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     addStop - start)
                     .count());
    printResults(subOut, totalThreads,
                 std::string("Subtraction Results"),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     subStop - addStop).count());
    printResults(multOut, totalThreads,
                 std::string("Multiplication Results"),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     multStop - subStop).count());
    printResults(modOut, totalThreads,
                 std::string("Modulus Results"),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     modStop - multStop).count());
    free(a);
    free(b);
    free(addOut);
    free(subOut);
    free(multOut);
    free(modOut);
}

void
processInput(std::vector<int> &totalThreadVec,
             std::vector<int> blockSizeVec,
             const int argc,
             char **argv)
{
    // read command line arguments
    if (argc >= 2)
    {
        printf("Total threads changed from %d to %d\n", totalThreadVec.at(0),
               atoi(argv[1]));
        totalThreadVec.at(0) = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        printf("Block size changed from %d to %d\n", blockSizeVec.at(0),
               atoi(argv[2]));
        blockSizeVec.at(0) = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        printf("Total threads changed from %d to %d\n", totalThreadVec.at(1),
               atoi(argv[3]));
        totalThreadVec.at(1) = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        printf("Block size changed from %d to %d\n",
               blockSizeVec.at(1),
               atoi(argv[4]));
        blockSizeVec.at(1) = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        printf("Total threads changed from %d to %d\n", totalThreadVec.at(2),
               atoi(argv[5]));
        totalThreadVec.at(2) = atoi(argv[5]);
    }
    if (argc >= 7)
    {
        printf("Block size changed from %d to %d\n",
               blockSizeVec.at(2),
               atoi(argv[6]));
        blockSizeVec.at(2) = atoi(argv[6]);
    }
}

int
main(int argc, char **argv)
{
    printDeviceInfo();

    std::vector<int> totalThreadVec;
    totalThreadVec.push_back(1 << 10);
    totalThreadVec.push_back(1 << 15);
    totalThreadVec.push_back(1 << 20);
    std::vector<int> blockSizeVec;
    blockSizeVec.push_back(64);
    blockSizeVec.push_back(128);
    blockSizeVec.push_back(256);

    processInput(totalThreadVec, blockSizeVec, argc, argv);

    // calculate all combinations of threads
    for (auto totalThread : totalThreadVec)
    {
        for (auto blockSize : blockSizeVec)
        {
            int numBlocks = totalThread / blockSize;

            // validate command line arguments
            if (totalThread % blockSize != 0)
            {
                ++numBlocks;
                totalThread = numBlocks * blockSize;

                printf(
                    "Warning: Total thread count is not evenly divisible by the block size\n");
                printf("The total number of threads will be rounded up to %d\n",
                       totalThread);
                printf("----------------------------------------------------\n");
            }

            printf("Performing operations on: #blocks (%d), block size (%d),  "
                   "total threads (%d)\n", numBlocks, blockSize, totalThread);
            printf("----------------------------------------------------\n");
            printf("----------------------------------------------------\n");
            doOperations(totalThread, numBlocks, blockSize);
        }
    }
}
