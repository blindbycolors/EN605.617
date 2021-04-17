#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

// OpenCL
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#define NUM_BUFFER_ELEMENTS 16
#define DEFAULT_PLATFORM    0
#define SUB_BUFFER_SIZE     4

#define CHECK_ERR(VAR_NAME, ERR_STR)                            \
    checkErr((VAR_NAME != CL_SUCCESS) ? VAR_NAME :              \
             (numPlatforms <= 0 ? -1 : CL_SUCCESS), ERR_STR);   \
    // End CHECK_ERR macro

#define GET_PLATFORM_INFO(PLAT_VAR, NUM_PLAT_VAR, VAR_ERR)                      \
    VAR_ERR = clGetPlatformIDs(0, NULL, &NUM_PLAT_VAR);                         \
    CHECK_ERR(VAR_ERR, "clGetPlatformIDs")                                      \
    PLAT_VAR = (cl_platform_id *)alloca(sizeof(cl_platform_id) * NUM_PLAT_VAR); \
    VAR_ERR = clGetPlatformIDs(NUM_PLAT_VAR, PLAT_VAR, NULL);                   \
    CHECK_ERR(VAR_ERR, "clGetPlatformIDs");                                     \
    // End GET_PLATFORM_INFO macro

#define GET_DEVICE_INFO(VAR_PLAT_NAME, VAR_DEV_NAME, VAR_COUNT, VAR_ERR)        \
    VAR_ERR = clGetDeviceIDs(VAR_PLAT_NAME[DEFAULT_PLATFORM],                   \
                             CL_DEVICE_TYPE_ALL, 0, NULL, &VAR_COUNT);          \
    if (VAR_ERR != CL_SUCCESS && VAR_ERR != CL_DEVICE_NOT_FOUND)                \
    {                                                                           \
        checkErr(VAR_ERR, "clGetDeviceIDs");                                    \
    }                                                                           \
    VAR_DEV_NAME = (cl_device_id *)alloca(sizeof(cl_device_id) * VAR_COUNT);    \
    VAR_ERR = clGetDeviceIDs(VAR_PLAT_NAME[DEFAULT_PLATFORM],                   \
                             CL_DEVICE_TYPE_ALL, VAR_COUNT,                     \
                             &VAR_DEV_NAME[DEFAULT_PLATFORM], NULL);            \
    checkErr(VAR_ERR, "clGetDeviceIDs");                                        \


#define CREATE_CONTEXT(VAR_CONTEXT, VAR_DEVICES, VAR_DEV_IDS, VAR_PLAT, VAR_ERR) \
    cl_context_properties contextProperties[] =                                  \
    {                                                                            \
        CL_CONTEXT_PLATFORM,                                                     \
        (cl_context_properties)VAR_PLAT[DEFAULT_PLATFORM],                       \
        0                                                                        \
    };                                                                           \
    VAR_CONTEXT = clCreateContext(contextProperties, VAR_DEVICES, VAR_DEV_IDS,   \
                                  NULL, NULL, &errNum);                          \
    checkErr(VAR_ERR, "clCreateContext");                                        \

// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void createBuffers(std::vector < cl_mem > &buffers, const cl_context& context,
                   const size_t numBuffers, const int bufferBytes,
                   int *inputOutput, bool runPaged)
{
    cl_int errNum;
    cl_mem buffer;

    // create a single buffer to cover all the input data
    if (runPaged)
    {
        buffer =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           bufferBytes, static_cast<void *>(inputOutput),
                           &errNum);
    }
    else
    {
        buffer =
            clCreateBuffer(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR |
                           CL_MEM_ALLOC_HOST_PTR,
                           bufferBytes, static_cast<void *>(inputOutput),
                           &errNum);
    }

    checkErr(errNum, "clCreateBuffer");
    buffers.push_back(buffer);

    // now for all devices other than the first create a sub-buffer
    for (unsigned int i = 0; i < numBuffers; i++)
    {
        // Create 2 x 2 region
        cl_buffer_region region =
        {
            SUB_BUFFER_SIZE *i *sizeof(int),
            SUB_BUFFER_SIZE *sizeof(int)
        };
        buffer = clCreateSubBuffer(buffers[0],
                                   CL_MEM_READ_WRITE,
                                   CL_BUFFER_CREATE_TYPE_REGION,
                                   &region,
                                   &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }
}

void createCommandQueues(cl_uint numDevices, std::vector<cl_command_queue> &queues,
                         const cl_context &context, cl_program program,
                         std::vector<cl_mem> buffers,
                         std::vector<cl_kernel> &kernels, cl_device_id *deviceIDs)
{
    cl_int subBufferSize = (cl_int)SUB_BUFFER_SIZE;
    auto bufferBytes = sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices;
    cl_int errNum;

    for (unsigned int i = 0; i < numDevices; i++)
    {
        cl_command_queue queue =
            clCreateCommandQueue(context, deviceIDs[i], 0, &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(program, "average", &errNum);
        checkErr(errNum, "clCreateKernel(average)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg(average)");

        errNum = clSetKernelArg(kernel, 1, sizeof(cl_int), &subBufferSize);
        checkErr(errNum, "clSetKernelArg(average)");

        kernels.push_back(kernel);
    }
}

void createEvents(std::vector<cl_event>&        events,
                  std::vector<cl_command_queue> queues,
                  std::vector<cl_kernel>        kernels)
{
    cl_int errNum;

    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = NUM_BUFFER_ELEMENTS;

        errNum =
            clEnqueueNDRangeKernel(queues[i], kernels[i], 1, NULL,
                                   (const size_t *)&gWI, (const size_t *)NULL,
                                   0, 0, &event);

        events.push_back(event);
    }
}

void print(cl_uint numDevices, int *inputOutput)
{
    // Display output in rows
    for (unsigned i = 0; i < numDevices; i++)
    {
        for (unsigned elems = i * NUM_BUFFER_ELEMENTS; elems < ((i + 1) * NUM_BUFFER_ELEMENTS); elems++)
        {
            std::cout << " " << inputOutput[elems];
        }

        std::cout << std::endl;
    }

    std::cout << "Program completed successfully" << std::endl;
}

void fillArray(int *inputOutput, cl_uint numDevices)
{
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput[i] = i;
    }
}

void runKernel(const cl_context& context,
               cl_device_id      *deviceIDs,
               cl_program        program,
               const cl_uint     numDevices,
               bool              runPaged)
{
    cl_int errNum;
    int *inputOutput;
    std::vector<cl_mem> buffers;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_event> events;

    size_t numBuffers = NUM_BUFFER_ELEMENTS / SUB_BUFFER_SIZE;
    auto bufferBytes = sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices;

    // create buffers and sub-buffers
    inputOutput = new int[NUM_BUFFER_ELEMENTS * numDevices];
    fillArray(inputOutput, numDevices);

    createBuffers(buffers, context, numBuffers, bufferBytes, inputOutput, runPaged);
    createCommandQueues(numDevices, queues, context, program, buffers,
                        kernels, deviceIDs);

    errNum = clEnqueueWriteBuffer(queues[0], buffers[0], CL_TRUE, 0, bufferBytes,
                                  (void *)inputOutput, 0, NULL, NULL);

    createEvents(events, queues, kernels);
    clWaitForEvents(events.size(), &events[0]);

    // Read back computed dat
    clEnqueueReadBuffer(queues[0], buffers[0], CL_TRUE, 0, bufferBytes,
                        (void *)inputOutput, 0, NULL, NULL);
    print(numDevices, inputOutput);
}

std::string loadProgram()
{
    std::ifstream srcFile("simple.cl");

    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(std::istreambuf_iterator<char>(srcFile),
                        (std::istreambuf_iterator<char>()));

    return srcProg;
}

void runPaged(const cl_context& context,
              cl_device_id      *deviceIDs,
              cl_program        program,
              const cl_uint     numDevices)
{
    bool runPaged = true;
    auto startTime = std::chrono::high_resolution_clock::now();

    runKernel(context, deviceIDs, program, numDevices, runPaged);
    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("Total Paged Memory Runtime: %f ms\n",
           std::chrono::duration<double, std::micro>(
               stopTime - startTime).count());
}

void runPinned(const cl_context& context,
               cl_device_id      *deviceIDs,
               cl_program        program,
               const cl_uint     numDevices)
{
    bool runPaged = false;
    auto startTime = std::chrono::high_resolution_clock::now();

    runKernel(context, deviceIDs, program, numDevices, runPaged);
    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("Total Pinned Memory Runtime: %f ms\n",
           std::chrono::duration<double, std::micro>(
               stopTime - startTime).count());
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char **argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id *platformIDs;
    cl_device_id *deviceIDs;
    cl_context context;
    cl_program program;

    // load the program
    auto prog = loadProgram();
    const char *src = prog.c_str();
    size_t length = prog.length();

    GET_PLATFORM_INFO(platformIDs, numPlatforms, errNum)
    GET_DEVICE_INFO(platformIDs, deviceIDs, numDevices, errNum);
    CREATE_CONTEXT(context, numDevices, deviceIDs, platformIDs, errNum)

    // Create program from source
    program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(program, numDevices, deviceIDs, "-I.", NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, deviceIDs[DEFAULT_PLATFORM], CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in OpenCL C source: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }

    runPinned(context, deviceIDs, program, numDevices);
    runPaged(context, deviceIDs, program, numDevices);

    return 0;
}