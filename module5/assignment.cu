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

void fillArrays(int *a, int *b, const int seed)
{
    for (auto i = 0; i < NUM_ELEMENTS; ++i)
    {
        a[i] = i;
        b[i] = randEngine(MIN, MAX, seed);
    }
}

void fillString(char *a, const int seed)
{
    printf("\nOriginal String:\t");
    for (auto i = 0; i < NUM_ELEMENTS; ++i)
    {
        auto randomChar = randEngine(MIN_PRINTABLE, MAX_PRINTABLE, seed);
        a[i] = randomChar;
        printf("%c", a[i]);
    }
    printf("\n");
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

void runPageable(const int offset, const int randomSeed)
{
    printf("Pageable memory."
           "\n-------------------------------------------------------------\n");
    HOST_ALLOCATE_PAGEABLE(aPageable, int, NUM_ELEMENTS);
    HOST_ALLOCATE_PAGEABLE(bPageable, int, NUM_ELEMENTS);
    HOST_ALLOCATE_PAGEABLE(toEncrypt, char, NUM_ELEMENTS);
    HOST_ALLOCATE_PAGEABLE(toDecrypt, char, NUM_ELEMENTS);

    fillArrays(aPageable, bPageable, randomSeed);

    hostAdd(aPageable, bPageable);
    hostSub(aPageable, bPageable);
    hostMult(aPageable, bPageable);
    hostMod(aPageable, bPageable);

    fillString(toEncrypt, randomSeed);
    auto encrypted = hostSharedEncrypt(toEncrypt, 3);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    auto decrypted = hostSharedDecrypt(toDecrypt, 3);
    printf("Encrypted String: %s\n", encrypted.c_str());
    printf("Decrypted String: %s\n", decrypted.c_str());

    fillString(toEncrypt, randomSeed);
    encrypted = hostConstantEncrypt(toEncrypt, 3);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    decrypted = hostConstantDecrypt(toDecrypt, 3);
    printf("Encrypted String: %s\n", encrypted.c_str());
    printf("Decrypted String: %s\n", decrypted.c_str());

    freeHostAlloc(aPageable, bPageable, toEncrypt, toDecrypt);
}

void runPinned(const int offset, const int randomSeed)
{
    printf("\n\nPinned memory."
           "\n-------------------------------------------------------------\n");
    HOST_ALLOCATE_PINNED(aPinned, int, NUM_ELEMENTS);
    HOST_ALLOCATE_PINNED(bPinned, int, NUM_ELEMENTS);
    HOST_ALLOCATE_PINNED(toEncrypt, char, NUM_ELEMENTS);
    HOST_ALLOCATE_PINNED(toDecrypt, char, NUM_ELEMENTS);

    fillArrays(aPinned, bPinned, randomSeed);

    hostAdd(aPinned, bPinned);
    hostSub(aPinned, bPinned);
    hostMult(aPinned, bPinned);
    hostMod(aPinned, bPinned);

    fillString(toEncrypt, randomSeed);
    auto encrypted = hostSharedEncrypt(toEncrypt, offset);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    auto decrypted = hostSharedDecrypt(toDecrypt, offset);
    printf("Encrypted String: %s\n", encrypted.c_str());
    printf("Decrypted String: %s\n", decrypted.c_str());

    fillString(toEncrypt, randomSeed);
    encrypted = hostConstantEncrypt(toEncrypt, offset);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    decrypted = hostConstantDecrypt(toDecrypt, offset);
    printf("Encrypted String: %s\n", encrypted.c_str());
    printf("Decrypted String: %s\n", decrypted.c_str());

    freePinned(aPinned, bPinned, toEncrypt, toDecrypt);
 }

int main(int argc, char **argv)
{
    int offset = 3;
    int seed = 100;
    if (argc >= 2)
    {
        offset = std::atol(argv[1]);
    }
    if(argc >= 3)
    {
        seed = std::atol(argv[2]);
    }

    printDeviceInfo();
    runPageable(offset, seed);
    runPinned(offset, seed);
    return 0;
}