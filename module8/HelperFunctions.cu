#include <iostream>
#include "HelperFunctions.cuh"


void generateSymmetricMatrix(const int size, double *matrix)
{
    double num;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < i + 1; j++)
        {
            num = randEngine<double>();
            matrix[i * size + j] = num;
            matrix[j * size + i] = num;
        }
    }
}

void printMultiplicationResults(float *deviceA, float *deviceB, float *deviceC,
                                const int aRows, const int aCols,
                                const int bRows, const int bCols)
{
    ALLOCATE_PAGEABLE_MEMORY(hostA, float, aRows * aCols);
    ALLOCATE_PAGEABLE_MEMORY(hostB, float, bRows * bCols);
    ALLOCATE_PAGEABLE_MEMORY(hostC, float, aRows * bCols);

    cudaMemcpy(hostA, deviceA, aRows * aCols * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hostB, deviceB, bRows * bCols * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hostC, deviceC, aRows * bCols * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "A = " << std::endl;
    printArray(hostA, aRows, aCols);
    std::cout << std::endl << "B = " << std::endl;
    printArray(hostB, bRows, bCols);
    std::cout << std::endl << "A * B = " << std::endl;
    printArray(hostC, aRows, bCols);
    freeHostAlloc(hostA, hostB);
};

void fillArrays(float *a, const int rows, const int cols)
{
    for (auto i = 0; i < rows; ++i)
    {
        for (auto j = 0; j < cols; ++j)
        {
            a[j * rows + i] = randEngine<float>();
        }
    }
}


void
gpuArrayMult(const float *a, const float *b, float *c, const int m, const int n,
             const int k)
{
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, b,
                ldb, beta, c, ldc);
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Kernel execution error during cublasSgemm.\n");
    }
    cublasDestroy(handle);
}


void findSymmetricMatrixEigenValues(double *d_A, double *d_W, const int m)
{
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status;
    const int lda = m;
    int lwork = 0;

    // step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    DEVICE_ALLOCATE(devInfo, int, 1);

    // step 2: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A,
                                                  lda, d_W, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    DEVICE_ALLOCATE(d_work, double, lwork);

    // step 3: compute
    cusolver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W,
                                       d_work, lwork, devInfo);
    checkCuda(cudaDeviceSynchronize());
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // free resources
    freeDeviceAlloc(devInfo, d_work);
    if (cusolverH)
    {
        cusolverDnDestroy(cusolverH);
    }
}

void printEigenResults(double *array, double *eigenVals, const int dim)
{
    printf("\nEigen Values:\n");
    for (int i = 0; i < dim; i++)
    {
        printf("eigenVals[%d] = %2.4f\n", i + 1, eigenVals[i]);
    }

    printf("\nArray after syevd:\n");
    printArray(array, dim, dim);
}

void computeEigenvalues(double *matrix, const int dim)
{
    // Copy the host data to device memory
    DEVICE_ALLOCATE(d_W, double, dim);
    DEVICE_ALLOCATE(d_A, double, dim * dim);
    ALLOCATE_PAGEABLE_MEMORY(eigenVals, double, dim);
    ALLOCATE_PINNED_MEMORY(array, double , dim * dim);

    checkCuda(cudaMemcpy(d_A, matrix, sizeof(double) * dim * dim,
                         cudaMemcpyHostToDevice));
    findSymmetricMatrixEigenValues(d_A, d_W, dim);

    checkCuda(cudaMemcpy(eigenVals, d_W, sizeof(double) * dim,
                         cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(array, d_A, sizeof(double) * dim * dim,
                         cudaMemcpyDeviceToHost));

    printf("Symmetric Input Matrix:\n");
    printArray(matrix, dim, dim);
    printEigenResults(array, eigenVals, dim);

    freeDeviceAlloc(d_W, d_A);
}

void runMatrixMult(float *a, float *b, const int aRows, const int aCols,
                   const int bRows, const int bCols)
{
    DEVICE_ALLOCATE(d_a, float, aRows * aCols);
    DEVICE_ALLOCATE(d_b, float, bRows * bCols);
    DEVICE_ALLOCATE(d_c, float, aRows * bCols);

    auto startTime = std::chrono::high_resolution_clock::now();
    fillArrays(a, aRows, aCols);
    fillArrays(b, bRows, bCols);
    checkCuda(cudaMemcpy(d_a, a, sizeof(float) * aRows * aCols,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, b, sizeof(float) * bRows * bCols,
                         cudaMemcpyHostToDevice));
    gpuArrayMult(d_a, d_b, d_c, aRows, bCols, bRows);

    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("\nRuntime: %f seconds\n",
           std::chrono::duration<double>(stopTime - startTime).count());

    printMultiplicationResults(d_a, d_b, d_c, aRows, aCols, bRows, bCols);
    printf("\n------------------------------------------------------------\n");

    freeDeviceAlloc(d_a, d_b, d_c);
}

void
runEigen(double *matrix, const int eigenDim)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    generateSymmetricMatrix(eigenDim, matrix);
    computeEigenvalues(matrix, eigenDim);
    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("\nRuntime: %f seconds\n",
           std::chrono::duration<double>(stopTime - startTime).count());

    printf("\n------------------------------------------------------------\n");
}
