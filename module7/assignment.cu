#include "HelperFunctions.cuh"
#include <string>
#include <random>

inline std::mt19937& generator(const int seed) {
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(seed);
    return gen;
}

// A function to generate integers in the range [min, max]
template<typename T>
T randEngine(T min, T max, const int seed) {
    std::uniform_int_distribution<T> dist(min, max);
    return dist(generator(seed));
}

void fillArrays(int *a, int *b, const int seed, const int numStreams)
{
    for (auto i = 0; i < numStreams; ++i)
    {
        a[i] = i;
        b[i] = randEngine(MIN, MAX, seed);
    }
}


void printDeviceInfo()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Compute Mode: %d\n", prop.computeMode);
        printf("  L2 Cache Size: %d\n", prop.l2CacheSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

void parseArgs(int argc, char **argv, int &seed, int &numStream,
               int &blockSize)
{
    int i = 1;
    while(i + 1 < argc)
    {
        if(std::string("--seed").compare(argv[i]) == 0)
        {
            seed = std::atoi(argv[i+1]);
        }
        else if(std::string("--num_stream").compare(argv[i]) == 0)
        {
            numStream = std::atoi(argv[i+1]);
        }
        else if(std::string("--block_size").compare(argv[i]) == 0)
        {
            blockSize = std::atoi(argv[i+1]);
        }

        i += 2;
    }

    if (blockSize != 256 || numStream != 2048)
    {
        if(numStream % blockSize != 0)
        {
            auto numBlocks = (numStream / blockSize) + 1;
            numStream = numBlocks * blockSize;
            printf(
                    "Warning: Total thread count is not evenly divisible by the block size\n");
            printf("The total number of threads will be rounded up to %d\n",
                   numStream);
            printf("----------------------------------------------------\n");
        }
    }
}

void runPinned(const int randomSeed, const int numStreams,
               const int numBlocks, std::vector<float> &times)
{
    ALLOCATE_PINNED_MEMORY(aPinned, int, numStreams);
    ALLOCATE_PINNED_MEMORY(bPinned, int, numStreams);

    fillArrays(aPinned, bPinned, randomSeed, numStreams);

    hostAdd(aPinned, bPinned, numStreams, numBlocks, times);
    hostSub(aPinned, bPinned, numStreams, numBlocks, times);
    hostMult(aPinned, bPinned, numStreams, numBlocks, times);
    hostMod(aPinned, bPinned, numStreams, numBlocks, times);

    freePinned(aPinned, bPinned);
}

void runPageable(const int randomSeed, const int numStreams,
                 const int numBlocks, std::vector<float> &times)
{
    ALLOCATE_PAGEABLE_MEMORY(aPageable, int, numStreams);
    ALLOCATE_PAGEABLE_MEMORY(bPageable, int, numStreams);

    fillArrays(aPageable, bPageable, randomSeed, numStreams);

    hostAdd(aPageable, bPageable, numStreams, numBlocks, times);
    hostSub(aPageable, bPageable, numStreams, numBlocks, times);
    hostMult(aPageable, bPageable, numStreams, numBlocks, times);
    hostMod(aPageable, bPageable, numStreams, numBlocks, times);

    freeHostAlloc(aPageable, bPageable);
}

void printTimeMetrics(const std::vector<float> times, const std::string& label)
{

    std::string spacing("       ");

    printf("%s", label.c_str());
    for(auto time : times)
    {
        printf("%f%s", time, spacing.c_str());
    }
    printf("\n");
}

void printBandwidthMetrics(const std::vector<float> times,
                           const std::string& label,
                           const int numStreams)
{

    std::string spacing("       ");

    printf("%s", label.c_str());
    for(auto time : times)
    {
        printf("%f%s", numStreams * 4 * 3 / time / 1e6, spacing.c_str());
    }
    printf("\n");
}

void printMetrics(const std::vector<float> pinnedTimes,
                  const std::vector<float> pagedTimes,
                  const int numStreams)
{
    printf("                ADD            SUB            MUL"
           "            MOD\n");
    printf
            ("========================================================="
             "=================\n");
    printf("Run times (ms)\n");
    printTimeMetrics(pinnedTimes, "PINNED MEM      ");
    printTimeMetrics(pagedTimes, "PAGED MEM       ");
    printf("\nBandwidth (GB/s)\n");
    printBandwidthMetrics(pinnedTimes, "PINNED MEM      ", numStreams);
    printBandwidthMetrics(pagedTimes, "PAGED MEM       ", numStreams);
}

int main(int argc, char **argv)
{
    int seed = 100;
    int blockSize = 256;
    int numStreams = 2048;
    std::vector<float> pinnedTimes, pagedTimes;

    parseArgs(argc, argv, seed, numStreams, blockSize);
    printDeviceInfo();


    int numBlocks = numStreams / blockSize;
    runPinned(seed, numStreams, numBlocks, pinnedTimes);
    runPageable(seed, numStreams, numBlocks, pagedTimes);

    printMetrics(pinnedTimes, pagedTimes, numStreams);
}