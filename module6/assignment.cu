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
    for (auto i = 0; i < KERNEL_LOOPS; ++i)
    {
        a[i] = i;
        b[i] = randEngine(MIN, MAX, seed);
    }
}

void fillString(char *a, const int seed)
{
#ifdef DEBUG
    printf("\nOriginal String:\t");
#endif
    for (auto i = 0; i < KERNEL_LOOPS; ++i)
    {
        auto randomChar = randEngine(MIN_PRINTABLE, MAX_PRINTABLE, seed);
        a[i] = randomChar;
#ifdef DEBUG
        printf("%c", a[i]);
#endif
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

void runPageableCipher(const int offset, const int randomSeed,
                       std::vector<float> &encryptRuntime,
                       std::vector<float> &decryptRuntime)
{
    HOST_ALLOCATE_PAGEABLE(toEncrypt, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(toDecrypt, char, KERNEL_LOOPS);

    fillString(toEncrypt, randomSeed);
    auto encrypted = hostSharedEncrypt(toEncrypt, offset, encryptRuntime);
    auto decrypted = hostSharedDecrypt(toDecrypt, offset, decryptRuntime);
#ifdef DEBUG
    printf("Shared Mem Encrypted String: %s\n", encrypted.c_str());
    printf("Shared Mem Decrypted String: %s\n", decrypted.c_str());
#endif

    fillString(toEncrypt, randomSeed);
    encrypted = hostConstantEncrypt(toEncrypt, offset, encryptRuntime);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    decrypted = hostConstantDecrypt(toDecrypt, offset, decryptRuntime);
#ifdef DEBUG
    printf("Constant Mem Encrypted String: %s\n", encrypted.c_str());
    printf("Constant Mem Decrypted String: %s\n", decrypted.c_str());
#endif

    fillString(toEncrypt, randomSeed);
    encrypted = hostRegisterEncrypt(toEncrypt, offset, encryptRuntime);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    decrypted = hostRegisterDecrypt(toDecrypt, offset, decryptRuntime);
#ifdef DEBUG
    printf("Register Mem Encrypted String: %s\n", encrypted.c_str());
    printf("Register Meme Decrypted String: %s\n", decrypted.c_str());
#endif

    freeHostAlloc(toEncrypt, toDecrypt);
}

void runPinnedCipher(const int offset, const int randomSeed,
                     std::vector<float> &encryptTime,
                     std::vector<float> &decryptTime)
{
    HOST_ALLOCATE_PINNED(toEncrypt, char, KERNEL_LOOPS);
    HOST_ALLOCATE_PINNED(toDecrypt, char, KERNEL_LOOPS);

    fillString(toEncrypt, randomSeed);
    auto encrypted = hostSharedEncrypt(toEncrypt, offset, encryptTime);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    auto decrypted = hostSharedDecrypt(toDecrypt, offset, decryptTime);
#ifdef DEBUG
    printf("Shared Mem Encrypted String: %s\n", encrypted.c_str());
    printf("Shared Mem Decrypted String: %s\n", decrypted.c_str());
#endif

    fillString(toEncrypt, randomSeed);
    encrypted = hostConstantEncrypt(toEncrypt, offset, encryptTime);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    decrypted = hostConstantDecrypt(toDecrypt, offset, decryptTime);
#ifdef DEBUG
    printf("Constant Mem Encrypted String: %s\n", encrypted.c_str());
    printf("Constant Mem Decrypted String: %s\n", decrypted.c_str());
#endif

    fillString(toEncrypt, randomSeed);
    encrypted = hostRegisterEncrypt(toEncrypt, offset, encryptTime);
    std::copy(encrypted.begin(), encrypted.end(), toDecrypt);
    decrypted = hostRegisterDecrypt(toDecrypt, offset, decryptTime);
#ifdef DEBUG
    printf("Register Mem Encrypted String: %s\n", encrypted.c_str());
    printf("Register Meme Decrypted String: %s\n", decrypted.c_str());
#endif

    freePinned( toEncrypt, toDecrypt);
}

void runPageable(const int randomSeed,
                 std::vector<float>& add,
                 std::vector<float>& sub,
                 std::vector<float>& mult,
                 std::vector<float>& mod)
{
    HOST_ALLOCATE_PAGEABLE(aPageable, int, KERNEL_LOOPS);
    HOST_ALLOCATE_PAGEABLE(bPageable, int, KERNEL_LOOPS);

    fillArrays(aPageable, bPageable, randomSeed);

    hostAdd(aPageable, bPageable, add);
    hostSub(aPageable, bPageable, sub);
    hostMult(aPageable, bPageable, mult);
    hostMod(aPageable, bPageable, mod);

    freeHostAlloc(aPageable, bPageable);
}

void runPinned(const int randomSeed,
               std::vector<float>& add,
               std::vector<float>& sub,
               std::vector<float>& mult,
               std::vector<float>& mod)
{
    HOST_ALLOCATE_PINNED(aPinned, int, KERNEL_LOOPS);
    HOST_ALLOCATE_PINNED(bPinned, int, KERNEL_LOOPS);

    fillArrays(aPinned, bPinned, randomSeed);

    hostAdd(aPinned, bPinned, add);
    hostSub(aPinned, bPinned, sub);
    hostMult(aPinned, bPinned, mult);
    hostMod(aPinned, bPinned, mod);

    freePinned(aPinned, bPinned);
}

void printTime(const std::vector<float> timeArr, const std::string& label)
{
    printf
    ("\n=================================================================="
     "========================================================================"
     "=================\n");
    printf("%s\t\t", label.c_str());
    auto count = 0;
    for(auto time : timeArr)
    {
        if (count == 2)
        {
            printf("   %f            ", time );
        }
        else if(count == 3)
        {
            printf("   %f          ", time);
        }
        else
        {
            printf("%f            ", time);
        }
        ++count;
    }
}

void printBandwidth(const std::vector<float> timeArr, const std::string& label)
{
    printf("\n=================================================================="
             "========================================================================"
             "=================\n");
    printf("%s\t\t", label.c_str());
    auto count = 0;
    for(auto time : timeArr)
    {
        if (count == 2)
        {
            printf("   %f            ",
                   KERNEL_LOOPS * 4 * 3 / time / 1e6);
        }
        else if(count == 3)
        {
            printf("   %f          ",
                   KERNEL_LOOPS * 4 * 3 / time / 1e6);
        }
        else
        {
            printf("%f            ", KERNEL_LOOPS * 4 * 3 / time / 1e6);
        }
        ++count;
    }
}

void printMetricsTimeTable(const std::vector<float> addTime,
                           const std::vector<float> subTime,
                           const std::vector<float> modTime,
                           const std::vector<float> multTime,
                           const std::vector<float> decryptTime,
                           const std::vector<float> encryptTime)
{
    printf("Runtime Metrics (ms):\n");
    printf("                Shared (Pageable)   Constant (Pageable)"
           "    Register (Pageable)    Shared(Pinned)"
           "    Constant (Pinned)   Register (Pinned)");
    printTime(addTime, "ADDITION");
    printTime(subTime, "SUBTRACTION");
    printTime(modTime, "MODULUS\t");
    printTime(multTime, "MULTIPLY");
    printTime(encryptTime, "ENCRYPT\t");
    printTime(decryptTime, "DECRYPT\t");
    printf("\n\n");
}

void printMetricsBandwidthTable(const std::vector<float> addTime,
                           const std::vector<float> subTime,
                           const std::vector<float> modTime,
                           const std::vector<float> multTime,
                           const std::vector<float> decryptTime,
                           const std::vector<float> encryptTime)
{
    printf("Bandwidth Metrics (GB/s):\n");
    printf("                Shared (Pageable)   Constant (Pageable)"
           "    Register (Pageable)    Shared(Pinned)"
           "    Constant (Pinned)   Register (Pinned)");
    printBandwidth(addTime, "ADDITION");
    printBandwidth(subTime, "SUBTRACTION");
    printBandwidth(modTime, "MODULUS    ");
    printBandwidth(multTime, "MULTIPLY");
    printBandwidth(encryptTime, "ENCRYPT    ");
    printBandwidth(decryptTime, "DECRYPT    ");
    printf("\n\n");
}

int main(int argc, char **argv)
{
    int offset = 3;
    int seed = 100;

    if (argc >= 2)
    {
        offset = std::atoi(argv[1]);
    }
    if(argc >= 3)
    {
        seed = std::atoi(argv[2]);
    }

    std::vector<float> addTime, subTime, multTime, modTime,
    encryptTime, decryptTime;

    printDeviceInfo();
    runPageable(seed, addTime, subTime, multTime, modTime);
    runPageableCipher(offset, seed, encryptTime, decryptTime);
    runPinned(seed, addTime, subTime, multTime, modTime);
    runPinnedCipher(offset, seed, encryptTime, decryptTime);

    printMetricsTimeTable(addTime, subTime, modTime, multTime, decryptTime,
                          encryptTime);
    printMetricsBandwidthTable(addTime, subTime, modTime, multTime,
                               decryptTime, encryptTime);

    return 0;
}

