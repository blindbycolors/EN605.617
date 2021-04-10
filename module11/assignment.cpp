#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
#define MIN 0
#define MAX 10

const unsigned int inputSignalWidth = 49;
const unsigned int inputSignalHeight = 49;
const unsigned int maskWidth = 7;
const unsigned int maskHeight = 7;
const unsigned int outputSignalWidth = 43;
const unsigned int outputSignalHeight = 43;

cl_uint mask[maskHeight][maskWidth] =
{
    { 1, 1, 1, 1, 1, 1, 1 },
    { 1, 2, 2, 2, 2, 2, 1 },
    { 1, 2, 3, 3, 3, 2, 1 },
    { 1, 2, 3, 4, 3, 2, 1 },
    { 1, 2, 3, 3, 3, 2, 1 },
    { 1, 2, 2, 2, 2, 2, 1 },
    { 1, 1, 1, 1, 1, 1, 1 }
};

cl_uint inputSignal[inputSignalHeight][inputSignalWidth];
cl_uint outputSignal[outputSignalHeight][outputSignalWidth];

//
// Macros
//
#define CHECK_CL_OBJECTS(VAR, ERROR_MSG)                                \
    if (VAR == NULL)                                                    \
    {                                                                   \
        std::cerr << ERROR_MSG << std::endl;                            \
        Cleanup(context, commandQueue, program, kernel);                \
        exit(1);                                                        \
    }                                                                   \
// End CHECK_CL_OBJECTS macro

#define CHECK_CL_STATUS(VAR, ERROR_MSG)                                 \
    if (VAR != CL_SUCCESS)                                              \
    {                                                                   \
        std::cerr << ERROR_MSG << std::endl;                            \
        exit(1);                                                        \
    }                                                                   \
// End CHECK_CL_STATUS macro

void
checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(const char *errInfo, const void *private_info,
                                 size_t cb, void *user_data)
{
    std::cout << "Error occured during context use: " << errInfo << std::endl;

    exit(1);
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel)
{
    if (commandQueue != 0) clReleaseCommandQueue(commandQueue);

    if (kernel != 0) clReleaseKernel(kernel);

    if (program != 0) clReleaseProgram(program);

    if (context != 0) clReleaseContext(context);
}

void CleanupMemObjs(cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0) clReleaseMemObject(memObjects[i]);
    }
}

void PrintInputSignal()
{
    std::cout << "===================================================="
                 "===================================================="
              << std::endl;

    for (auto i = 0; i < inputSignalHeight; ++i)
    {
        for (auto j = 0; j < inputSignalWidth; ++j)
        {
            std::cout << inputSignal[i][j] << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl
              << "===================================================="
              << "===================================================="
              << std::endl;
}

void PrintOutputSignal()
{
    std::cout << "===================================================="
                 "===================================================="
              << std::endl;

    for (auto i = 0; i < outputSignalHeight; ++i)
    {
        for (auto j = 0; j < outputSignalWidth; ++j)
        {
            std::cout << outputSignal[i][j] << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl
              << "===================================================="
              << "===================================================="
              << std::endl;
}

std::mt19937& generator(const int seed)
{
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(seed);

    return gen;
}

// A function to generate integers in the range [min, max]
unsigned int randEngine(unsigned int min, unsigned int max)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::uniform_int_distribution<unsigned int> dist(min, max);

    return dist(generator(seed));
}

void fillSignal()
{
    for (auto i = 0; i < inputSignalHeight; ++i)
    {
        for (auto j = 0; j < inputSignalWidth; ++j)
        {
            inputSignal[i][j] = randEngine(MIN, MAX);
        }
    }
}

void createProgram(cl_program & program,
                   cl_context & context,
                   cl_uint &    numDevices,
                   cl_device_id *deviceIDs)
{
    cl_int errNum;

    std::ifstream srcFile("Convolution.cl");

    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

    std::string srcProg(std::istreambuf_iterator<char>(srcFile),
                        (std::istreambuf_iterator<char>()));

    const char *src = srcProg.c_str();
    size_t length = srcProg.length();

    // Create program from source
    program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(program, numDevices, deviceIDs, NULL, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }
}

void CreatePagedMemObjects(cl_context context, cl_mem &inputSignalBuffer,
                           cl_mem &outputSignalBuffer, cl_mem &maskBuffer)
{
    cl_int errNum;

    inputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
        static_cast<void *>(inputSignal),
        &errNum);
    checkErr(errNum, "clCreateBuffer(inputSignal)");

    maskBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * maskHeight * maskWidth,
        static_cast<void *>(mask),
        &errNum);
    checkErr(errNum, "clCreateBuffer(mask)");

    outputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer(outputSignal)");
}

void CreatePinnedMemObjects(cl_context context, cl_mem &inputSignalBuffer,
                            cl_mem &outputSignalBuffer, cl_mem &maskBuffer)
{
    cl_int errNum;

    inputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
        sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
        static_cast<void *>(inputSignal),
        &errNum);
    checkErr(errNum, "clCreateBuffer(inputSignal)");

    maskBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
        sizeof(cl_uint) * maskHeight * maskWidth,
        static_cast<void *>(mask),
        &errNum);
    checkErr(errNum, "clCreateBuffer(mask)");

    outputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer(outputSignal)");
}

void RunProgram(cl_command_queue &commandQueue, cl_kernel& kernel,
                cl_mem &inputSignalBuffer, cl_mem &outputSignalBuffer,
                cl_mem &maskBuffer)
{
    cl_int errNum;

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
    checkErr(errNum, "clSetKernelArg");

    const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2] = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");

    errNum = clEnqueueReadBuffer(commandQueue, outputSignalBuffer, CL_TRUE, 0,
                                 sizeof(cl_uint) * outputSignalHeight *
                                 outputSignalHeight,
                                 outputSignal, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueReadBuffer");
}

void RunPageable(cl_context &context, cl_command_queue &commandQueue,
                 cl_kernel &kernel, cl_program &program)
{
    cl_mem inputSignalBuffer;
    cl_mem outputSignalBuffer;
    cl_mem maskBuffer;
    auto startTime = std::chrono::high_resolution_clock::now();

    CreatePagedMemObjects(context, inputSignalBuffer,
                          outputSignalBuffer, maskBuffer);
    RunProgram(commandQueue, kernel, inputSignalBuffer, outputSignalBuffer,
               maskBuffer);

    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("Total Paged Memory Runtime: %f ms\n",
           std::chrono::duration<double, std::micro>(
               stopTime - startTime).count());
}

void RunPinned(cl_context &context, cl_command_queue &commandQueue,
               cl_kernel &kernel, cl_program &program)
{
    cl_mem inputSignalBuffer;
    cl_mem outputSignalBuffer;
    cl_mem maskBuffer;
    auto startTime = std::chrono::high_resolution_clock::now();

    CreatePinnedMemObjects(context, inputSignalBuffer,
                           outputSignalBuffer, maskBuffer);
    RunProgram(commandQueue, kernel, inputSignalBuffer, outputSignalBuffer,
               maskBuffer);

    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("Total Pinned Memory Runtime: %f ms\n",
           std::chrono::duration<double, std::micro>(
               stopTime - startTime).count());
}

///
//	main()
//
int main(int argc, char **argv)
{
    cl_context context = NULL;
    cl_command_queue commandQueue;
    cl_program program;
    cl_uint numPlatforms, numDevices;
    cl_platform_id *platformIDs;
    cl_device_id *deviceIDs = NULL;
    cl_kernel kernel;
    cl_int errNum, i;

    fillSignal();
    #ifdef DEBUG
    PrintInputSignal();
    #endif

    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    platformIDs = (cl_platform_id *)alloca(
        sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    for (i = 0; i < numPlatforms; i++)
    {
        errNum = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL,
                                &numDevices);

        if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
        {
            checkErr(errNum, "clGetDeviceIDs");
        }
        else if (numDevices > 0)
        {
            deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
            errNum = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU,
                                    numDevices, &deviceIDs[0], NULL);
            checkErr(errNum, "clGetDeviceIDs");
            break;
        }
    }

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[i], 0
    };

    context = clCreateContext(contextProperties, numDevices, deviceIDs,
                              &contextCallback, NULL, &errNum);
    checkErr(errNum, "clCreateContext");

    createProgram(program, context, numDevices, deviceIDs);
    // Create kernel object
    kernel = clCreateKernel(program, "convolve", &errNum);
    checkErr(errNum, "clCreateKernel");

    //Create the command queue
    // Pick the first device and create command queue.
    commandQueue = clCreateCommandQueue(context, deviceIDs[0], 0, &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    RunPageable(context, commandQueue, kernel, program);
    #ifdef DEBUG
    std::cout << "Results after running paged memory: " << std::endl;
    PrintOutputSignal();
    #endif
    RunPinned(context, commandQueue, kernel, program);
    #ifdef DEBUG
    std::cout << "Results after running pinned memory: " << std::endl;
    PrintOutputSignal();
    #endif

    return 0;
}