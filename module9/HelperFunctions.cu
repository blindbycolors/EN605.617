#include "HelperFunctions.cuh"
#include <chrono>


void print(thrust::host_vector<int> vec, const std::string& label)
{
    printf("%s: ", label.c_str());
    for (auto curr : vec)
    {
        printf("%d ", curr);
    }
    printf("\n");
}

void runOperations(int * hostA,
                   int * hostB,
                   const int N,
                   thrust::host_vector<int> &hAdd,
                   thrust::host_vector<int> &hSub,
                   thrust::host_vector<int> &hMul,
                   thrust::host_vector<int> &hMod)
{
    std::chrono::time_point<std::chrono::system_clock> startTime;
    std::chrono::time_point<std::chrono::system_clock> endTime;
    std::chrono::duration<double> duration;

    // Copy from host to device
    startTime = std::chrono::high_resolution_clock::now();
    thrust::device_vector<int> dVectorA (hostA, hostA + N);
    thrust::device_vector<int> dVectorB (hostB, hostB + N);
    endTime = std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    printf("\tData Transfer Runtime: %ld us\n",
           std::chrono::duration_cast<
                   std::chrono::microseconds>(duration).count());

    // Do Math Operations
    DO_OPERATION(plus, dAdd, hAdd, "ADD");
    DO_OPERATION(minus, dMinus, hSub, "SUBTRACT");
    DO_OPERATION(multiplies, dMul, hMul, "MULTIPLY");
    DO_OPERATION(modulus, dMod, hMod, "MOD");
}

void printAll(int *hA,
              int *hB,
              int N,
              thrust::host_vector<int> add,
              thrust::host_vector<int> sub,
              thrust::host_vector<int> mul,
              thrust::host_vector<int> mod)
{
    printf("\nOperation Results:\n");
    thrust::host_vector<int> hVecA(hA, hA + N);
    thrust::host_vector<int> hVecB(hB, hB + N);
    print(hVecA, "A");
    print(hVecB, "B");
    print(add, "A + B");
    print(sub, "A - B");
    print(mul, "A * B");
    print(mod, "A mod B");
}
