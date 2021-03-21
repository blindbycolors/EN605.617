#include "HelperFunctions.cuh"
#include <iostream>
#include <cstdio>
#include <cstdlib>
void
runPageable(const int aRows, const int aCols, const int bRows, const int bCols,
            const int eigenDim)
{
    ALLOCATE_PAGEABLE_MEMORY(a, float, aRows * aCols);
    ALLOCATE_PAGEABLE_MEMORY(b, float, bRows * bCols);
    ALLOCATE_PAGEABLE_MEMORY(matrix, double, eigenDim * eigenDim);

    printf("Pageable Memory Matrix Mulitplication\n");
    runMatrixMult(a, b, aRows, aCols, bRows, bCols);

    printf("Pageable Eigen Value computation\n");
    runEigen(matrix, eigenDim);

    freeHostAlloc(a, b, matrix);
}


void
runPinned(const int aRows, const int aCols, const int bRows, const int bCols,
          const int eigenDim)
{

    ALLOCATE_PINNED_MEMORY(a, float, aRows * aCols);
    ALLOCATE_PINNED_MEMORY(b, float, bRows * bCols);
    ALLOCATE_PINNED_MEMORY(matrix, double, eigenDim * eigenDim);

    printf("Pinned Memory Matrix Mulitplication\n");
    runMatrixMult(a, b, aRows, aCols, bRows, bCols);

    printf("Pinned Eigen Value computation\n");
    runEigen(matrix, eigenDim);

    freePinned(a, b, matrix);
}


void parseArgs(int argc, char **argv, int &aRows, int &aCols,
               int &bRows, int &bCols, int &eigenDim)
{
    int i = 1;
    while(i + 1 < argc)
    {
        if(std::string("--a_row").compare(argv[i]) == 0)
        {
            aRows = std::atoi(argv[i+1]);
        }
        else if(std::string("--a_col").compare(argv[i]) == 0)
        {
            aCols = std::atoi(argv[i+1]);
        }
        else if(std::string("--b_row").compare(argv[i]) == 0)
        {
            bRows = std::atoi(argv[i+1]);
        }
        else if(std::string("--b_col").compare(argv[i]) == 0)
        {
            bCols = std::atoi(argv[i+1]);
        }
        else if(std::string("--eigen").compare(argv[i]) == 0)
        {
            eigenDim = std::atoi(argv[i+1]);
        }

        i += 2;
    }

    if (aCols != bRows)
    {
        fprintf(stderr, "Error: Number of rows from first matrix (%d) "
                        "must equal number of number of cols from second "
                        "matrix (%d)"
                        "\nSetting them equal.", aCols, bRows);
        bCols = aRows;
    }
}

int main(int argc, char **argv)
{
    int aRows, aCols, bRows, bCols, eigenDim;
    aRows = 1;
    aCols = 5;
    bRows = 5;
    bCols = 3;
    eigenDim = 3;

    parseArgs(argc, argv, aRows, aCols, bRows, bCols, eigenDim);

    printf("\n------------------------------------------------------------\n");
    runPageable(aRows, aCols, bRows, bCols, eigenDim);
    printf("\n------------------------------------------------------------\n");
    runPinned(aRows, aCols, bRows, bCols, eigenDim);

    cudaDeviceReset();
    return 0;
}
