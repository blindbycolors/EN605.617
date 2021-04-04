//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
const int ARRAY_SIZE = 1000;

//
// Macros
//
#define CHECK_CL_OBJECTS(VAR, ERROR_MSG)                                \
    if (VAR == NULL)                                                    \
    {                                                                   \
        std::cerr << ERROR_MSG << std::endl;                            \
        Cleanup(context, commandQueue, program, kernel);    \
        exit(1);                                                       \
    }                                                                   \
// End CHECK_CL_OBJECTS macro

#define CHECK_CL_STATUS(VAR, ERROR_MSG)                                 \
    if (VAR != CL_SUCCESS)                                              \
    {                                                                   \
        std::cerr << ERROR_MSG << std::endl;                            \
        exit(1);                                                       \
    }                                                                   \
// End CHECK_CL_STATUS macro

#define RUN_KERNEL(FUNC, LABEL)                                         \
    startTime = std::chrono::high_resolution_clock::now();              \
    FUNC(commandQueue, context, kernel, program, memObjects, result);   \
    stopTime = std::chrono::high_resolution_clock::now();               \
    printf("%s Runtime: %f ms\n", LABEL.c_str(),                        \
           std::chrono::duration<double, std::micro>                    \
               (stopTime - startTime).count());                         \
    PrintResults(result);                                               \
// End RUN_KERNEL macro

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select first OpenCL platform to run on
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);

    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0
    };

    context = clCreateContextFromType(contextProperties,
                                      CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);

        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceBufferSize,
                              devices,
                              NULL);

    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // Choose first available OpenCL device
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device,
                         const char *fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);

    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&srcStr,
                                        NULL, NULL);

    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreatePagedMemObjects(cl_context context, cl_mem memObjects[3],
                           float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

bool CreatePinnedMemObjects(cl_context context, cl_mem memObjects[3],
                            float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR |
                                   CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR |
                                   CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
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

//
// Print the mathematical function results
//
void PrintResults(float *results)
{
#ifdef DEBUG
    std::cout << "===================================================="
                 "===================================================="
              << std::endl;

    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        std::cout << results[i] << " ";
    }

    std::cout << std::endl
              << "===================================================="
              << "===================================================="
              << std::endl;
#endif
}

//
// Fill intial arrays
//
void FillArray(float *a, float *b)
{
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }
}

void RunProgram(cl_command_queue &commandQueue,
                cl_kernel &kernel,  cl_mem memObjects[3],
                float *result)
{
    cl_int errNum;

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    CHECK_CL_STATUS(errNum, "Error setting kernel arguments");

    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    CHECK_CL_STATUS(errNum, "Error queuing kernel for execution");

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    CHECK_CL_STATUS(errNum, "Error reading result buffer");
}

void RunAdd(cl_command_queue &commandQueue, cl_context &context,
            cl_kernel &kernel, cl_program &program, cl_mem memObjects[3],
            float *result)
{
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "add_kernel", NULL);
    CHECK_CL_OBJECTS(kernel, "Failed to create kernel");

    RunProgram(commandQueue, kernel, memObjects, result);
}

void RunSub(cl_command_queue &commandQueue, cl_context &context,
            cl_kernel &kernel, cl_program &program, cl_mem memObjects[3],
            float *result)
{
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "sub_kernel", NULL);
    CHECK_CL_OBJECTS(kernel, "Failed to create kernel");

    RunProgram(commandQueue, kernel, memObjects, result);
}

void RunMult(cl_command_queue &commandQueue, cl_context &context,
             cl_kernel &kernel, cl_program &program, cl_mem memObjects[3],
             float *result)
{
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "mult_kernel", NULL);
    CHECK_CL_OBJECTS(kernel, "Failed to create kernel");

    RunProgram(commandQueue, kernel, memObjects, result);
}

void RunDiv(cl_command_queue &commandQueue, cl_context &context,
            cl_kernel &kernel, cl_program &program, cl_mem memObjects[3],
            float *result)
{
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "div_kernel", NULL);
    CHECK_CL_OBJECTS(kernel, "Failed to create kernel");

    RunProgram(commandQueue, kernel, memObjects, result);
}

void RunPow(cl_command_queue &commandQueue, cl_context &context,
            cl_kernel &kernel, cl_program &program, cl_mem memObjects[3],
            float *result)
{
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "pow_kernel", NULL);
    CHECK_CL_OBJECTS(kernel, "Failed to create kernel");

    RunProgram(commandQueue, kernel, memObjects, result);
}

void RunPageable(cl_context &context, cl_command_queue &commandQueue,
                 cl_kernel &kernel, cl_program &program, cl_device_id device,
                 float *a, float *b)
{
    float result[ARRAY_SIZE];
    cl_mem memObjects[3];
    auto pageableStart = std::chrono::high_resolution_clock::now();

    if (!CreatePagedMemObjects(context, memObjects, a, b))
    {
        Cleanup(context, commandQueue, program, kernel);
        exit(1);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> stopTime;
    RUN_KERNEL(RunAdd, std::string("Add"))
    RUN_KERNEL(RunSub, std::string("Sub"))
    RUN_KERNEL(RunMult, std::string("Mult"))
    RUN_KERNEL(RunDiv, std::string("DIV"))
    RUN_KERNEL(RunPow, std::string("POW"))

    auto pageableStop = std::chrono::high_resolution_clock::now();
    printf("Total Paged Memroy Runtime: %f ms\n",
           std::chrono::duration<double, std::micro>
               (pageableStop - pageableStart).count());

    CleanupMemObjs(memObjects);
}

void RunPinned(cl_context &context, cl_command_queue &commandQueue,
               cl_kernel &kernel, cl_program &program, cl_device_id device,
               float *a, float *b)
{
    float result[ARRAY_SIZE];
    cl_mem memObjects[3];
    auto pageableStart = std::chrono::high_resolution_clock::now();

    if (!CreatePinnedMemObjects(context, memObjects, a, b))
    {
        Cleanup(context, commandQueue, program, kernel);
        exit(1);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> stopTime;
    RUN_KERNEL(RunAdd, std::string("Add"))
    RUN_KERNEL(RunSub, std::string("Sub"))
    RUN_KERNEL(RunMult, std::string("Mult"))
    RUN_KERNEL(RunDiv, std::string("DIV"))
    RUN_KERNEL(RunPow, std::string("POW"))

    auto pageableStop = std::chrono::high_resolution_clock::now();
    printf("Total Pinned Memory Runtime: %f ms\n",
           std::chrono::duration<double, std::micro>
               (pageableStop - pageableStart).count());

    CleanupMemObjs(memObjects);
}

///
//	main()
//
int main(int argc, char **argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];

    // Create an OpenCL context on first available platform
    context = CreateContext();

    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available on the created context
    commandQueue = CreateCommandQueue(context, &device);
    CHECK_CL_OBJECTS(commandQueue, "Erorr intializing command_queue");

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "HelloWorld.cl");
    CHECK_CL_OBJECTS(program, "Error initializing cl_program");

    FillArray(a, b);
    RunPageable(context, commandQueue, kernel, program, device, a, b);
    RunPinned(context, commandQueue, kernel, program, device, a, b);

    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel);

    return 0;
}