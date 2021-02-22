#include "operations.cuh"
#include "assignment.cuh"
#include <vector>
#include <algorithm>
#include <random>



int main(int argc, char **argv)
{
    if (argc >= 2)
    {
        auto program = std::string(argv[1]);
        if (program.compare("--do_opt") == 0)
        {
            doMathOperations(argc, argv);
        } else if (program.compare("--cipher") == 0)
        {
            doCipher(argc, argv);
        }
    } else
    {
        printf("No program specified. Performing all operations and cipher\n");
        doMathOperations(argc, argv);
        doCipher(argc, argv);
    }
}

std::vector<std::pair<int, int>> parseOperations(int argc, char **argv)
{
    // Parse inputs as BLOCK_SIZE TOTAL_THREADS
    std::vector<int> blockSizes;
    std::vector<int> totalThreads;
    // Vector containing unique pairs of (TOTAL_THREADS, BLOCK_SIZE)
    std::vector<std::pair<int, int>> uniquePairs;

    parseMathArgv(argc, argv, totalThreads, blockSizes);

    for (auto i = 0; i < totalThreads.size(); ++i)
    {
        for (auto j = 0; j < blockSizes.size(); ++j)
        {
            auto changedTotal = false;
            auto numBlocks = totalThreads.at(i) / blockSizes.at(j);
            auto currTotal = totalThreads.at(i);
            if (currTotal % blockSizes.at(j) != 0)
            {
                ++numBlocks;
                currTotal = numBlocks * blockSizes.at(j);
                changedTotal = true;
            }

            auto exists = std::find(uniquePairs.begin(), uniquePairs.end(),
                                    std::make_pair(currTotal,
                                                   blockSizes.at(j)));

            if (exists == uniquePairs.end())
            {
                uniquePairs.push_back(
                        std::make_pair(currTotal, blockSizes.at(j)));
                if (changedTotal)
                {
                    printWarning(totalThreads.at(i), currTotal,
                                 blockSizes.at(j));
                }
            }
        }
    }

    return uniquePairs;
}

void printWarning(const int origTotal, const int newTotal, const int blockSize)
{
    printf("Warning: Total thread count (%d) is not evenly "
           "divisible by the block size (%d)\n", origTotal, blockSize);
    printf("The total number of threads will be rounded up to %d\n", newTotal);
    printf("----------------------------------------------------\n");
}

void
parseMathArgv(const int argc, char **pString, std::vector<int> &totalThreads,
              std::vector<int> &blockSizes)
{
    for (auto i = 2; i < argc; ++i)
    {
        if (i % 2 == 0)
        {
            totalThreads.push_back(atoi(pString[i]));
        } else
        {
            blockSizes.push_back(atoi(pString[i]));
        }
    }
}

void doMathOperations(int argc, char **argv)
{
    auto uniqueCombos = parseOperations(argc, argv);
    // Default operations if there were no args provided
    if (uniqueCombos.empty())
    {
        int totalThreads = 1 << 20;
        int blockSize = 512;
        int numBlocks = totalThreads / blockSize;

        printf("Completing operations with default values:"
               "Total Threads (%d), Block Size (%d)\n", totalThreads, blockSize);

        // Allocate pageable memory
        HOST_ALLOCATE_PAGEABLE(totalThreads, aPageable, int)
        HOST_ALLOCATE_PAGEABLE(totalThreads, bPageable, int);
        // Allocate pinned memory
        HOST_ALLOCATE_PINNED(totalThreads, aPinned, int);
        HOST_ALLOCATE_PINNED(totalThreads, bPinned, int);

        // fill arithmetic array with values
        fillArrays(aPageable, bPageable, totalThreads);
        fillArrays(aPinned, bPinned, totalThreads);

        printf("Pageable Memory Metrics:\n");
        DO_OPERATIONS(aPageable, bPageable, hostAdd, addPageableDur,
                      "Addition");
        DO_OPERATIONS(aPageable, bPageable, hostSubtract, subPageableDur,
                      "Subtraction");
        DO_OPERATIONS(aPageable, bPageable, hostMultiply, multPageableDur,
                      "Multiplication");
        DO_OPERATIONS(aPageable, bPageable, hostMod, modPageableDur, "Modulus");

        printf("\nPinned Memory Metrics:\n");
        DO_OPERATIONS(aPinned, bPinned, hostAdd, addPinnedDur, "Addition");
        DO_OPERATIONS(aPinned, bPinned, hostSubtract, subPinnedDur,
                      "Subtraction");
        DO_OPERATIONS(aPinned, bPinned, hostMultiply, multPinnedDur,
                      "Multiplication")
        DO_OPERATIONS(aPinned, bPinned, hostMod, modPinnedDur, "Modulus");

        FREE_PINNED(aPinned, bPinned);
        FREE_PAGEABLE(aPageable, bPageable)

        printf("----------------------------------------------------\n");
    } else
    {
        doCombos(uniqueCombos);
    }
}

void doCombos(const std::vector<std::pair<int, int>> uniqueCombos)
{
    int totalThreads, blockSize, numBlocks;

    for (auto combo : uniqueCombos)
    {
        totalThreads = combo.first;
        blockSize = combo.first;
        numBlocks = totalThreads / blockSize;

        printf("Completing operations with input values: Total Threads (%d)"
               ", Block Size (%d)\n", totalThreads, blockSize);

        // Allocate pageable memory
        HOST_ALLOCATE_PAGEABLE(totalThreads, aPageable, int)
        HOST_ALLOCATE_PAGEABLE(totalThreads, bPageable, int);
        // Allocate pinned memory
        HOST_ALLOCATE_PINNED(totalThreads, aPinned, int);
        HOST_ALLOCATE_PINNED(totalThreads, bPinned, int);

        DO_OPERATIONS(aPageable, bPageable, hostAdd, addPageableDur,
                      "Addition");
        DO_OPERATIONS(aPageable, bPageable, hostSubtract, subPageableDur,
                      "Subtraction");
        DO_OPERATIONS(aPageable, bPageable, hostMultiply, multPageableDur,
                      "Multiplication");
        DO_OPERATIONS(aPageable, bPageable, hostMod, modPageableDur, "Modulus");

        printf("\nPinned Memory Metrics:\n");
        DO_OPERATIONS(aPinned, bPinned, hostAdd, addPinnedDur, "Addition");
        DO_OPERATIONS(aPinned, bPinned, hostSubtract, subPinnedDur,
                      "Subtraction");
        DO_OPERATIONS(aPinned, bPinned, hostMultiply, multPinnedDur,
                      "Multiplication")
        DO_OPERATIONS(aPinned, bPinned, hostMod, modPinnedDur, "Modulus");

        FREE_PINNED(aPinned, bPinned);
        FREE_PAGEABLE(aPageable, bPageable);

        printf("----------------------------------------------------\n");
    }
}

void doCipher(int argc, char **argv)
{
    auto offset = 3;
    auto strLength = 64;
    auto blockSize = 32;
    auto numBlocks = strLength / blockSize;

    if (argc == 4)
    {
        //do default example
        offset = atoi(argv[2]);
        strLength = atoi(argv[3]);

        numBlocks = strLength / blockSize;

        if (strLength % blockSize != 0)
        {
            ++numBlocks;
            printWarning(strLength, blockSize * numBlocks, blockSize);
            strLength = blockSize * strLength;
        }
    }

    HOST_ALLOCATE_PAGEABLE(strLength, aPageable, char);
    HOST_ALLOCATE_PAGEABLE(strLength, bPageable, char);
    HOST_ALLOCATE_PINNED(strLength, aPinned, char);
    HOST_ALLOCATE_PINNED(strLength, bPinned, char);

    printf("Original String:\t");
    for (auto i = 0; i < strLength; ++i)
    {
        auto randomChar = rand() % (MAX_PRINTABLE - MIN_PRINTABLE) +
                          MIN_PRINTABLE;
        aPageable[i] = randomChar;
        aPinned[i] = randomChar;
        printf("%c", aPageable[i]);
    }

    printf("\nPageable Memory Cipher:\n");
    auto encrypted = hostEncrypt(numBlocks, blockSize, aPageable,
                                 strLength, offset);
    std::copy(encrypted.begin(), encrypted.end(), bPageable);
    auto decrypted = hostDecrypt(numBlocks, blockSize, bPageable,
                                 strLength, offset);

    printf("\nPinned Memory Cipher:\n");
    hostEncrypt(numBlocks, blockSize, aPinned, strLength, offset);
    std::copy(encrypted.begin(), encrypted.end(), bPinned);
    hostDecrypt(numBlocks, blockSize, bPinned, strLength, offset);

    printf("\nEncrypted:\t\t%s\n", encrypted.c_str());
    printf("\nDecrypted:\t\t%s\n", decrypted.c_str());
}

void fillArrays(int *a, int *b, const int totalThreads)
{
    for (auto i = 0; i < totalThreads; ++i)
    {
        a[i] = i;
        b[i] = rand() % MAX + MIN;
    }
}
