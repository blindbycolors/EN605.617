#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <array>
#include <algorithm>
#include <numeric>
#include <utility>

// OpenCL
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#define DEFAULT_PLATFORM 0
#define BUFFER_SIZE      1024

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow()   std::chrono::high_resolution_clock::now()

template<typename F, typename ... Args>
void funcTime(F func, Args&& ... args)
{
    TimeVar t1 = timeNow();

    func(std::forward<Args>(args)...);
    std::cout << "Runtime (ns): " << duration(timeNow() - t1) << std::endl;
}

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

std::vector<std::string> parseArgs(int argc, char **argv)
{
    std::vector<std::string> cmds;

    for (auto i = 1; i < argc; ++i)
    {
        cmds.push_back(argv[i]);
    }

    return cmds;
}

std::vector<cl_kernel> createKernels(std::vector<std::string> commands,
                                     const cl_program&        program)
{
    cl_int errNum;
    std::vector<cl_kernel> kernels;

    for (auto command : commands)
    {
        if (command.compare("square") || command.compare("cube") ||
            command.compare("bitwiseDouble") || command.compare("bitwiseHalf"))
        {
            cl_kernel kernel = clCreateKernel(program, command.c_str(), &errNum);
            std::string errStr = "clCreateKernel(" + command + ")";
            checkErr(errNum, errStr.c_str());
            kernels.push_back(kernel);
        }
        else
        {
            std::cerr << "Unknown command provided. Skipping: "
                      << command << std::endl;
        }
    }

    return kernels;
}

std::string loadProgramStr()
{
    std::ifstream srcFile("simple.cl");

    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(std::istreambuf_iterator<char>(srcFile),
                        (std::istreambuf_iterator<char>()));

    return srcProg;
}

void print(cl_uint numDevices, int *inputOutput)
{
    // Display output in rows
    std::cout << "Output after commands: " << std::endl;

    for (unsigned i = 0; i < numDevices; i++)
    {
        for (unsigned elems = i * BUFFER_SIZE;
             elems < ((i + 1) * BUFFER_SIZE); elems++)
        {
            std::cout << " " << inputOutput[elems];
        }

        std::cout << std::endl;
    }

    std::cout << "Program completed successfully" << std::endl;
}

void executeCommands(const cl_context &context, const cl_device_id *deviceIDs,
                     const std::vector<cl_kernel> &kernels, cl_uint numDevices,
                     int *inputOutput)
{
    cl_int errNum;
    auto bufferBytes = sizeof(int) * BUFFER_SIZE * numDevices;
    size_t gWI = BUFFER_SIZE;
    std::vector<cl_event> events;

    // Create buffer for device
    cl_mem buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bufferBytes, NULL, &errNum);

    checkErr(errNum, "clCreateBuffer");

    //Create command queues
    cl_command_queue queue =
        clCreateCommandQueue(context, deviceIDs[0], 0, &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    // Copy data from host to device
    // Write input data
    errNum =
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, bufferBytes,
                             (void *)inputOutput, 0, NULL, NULL);

    // Enqueue each kernel
    for (auto& kernel : kernels)
    {
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer);
        checkErr(errNum, "clSetKernelArg");
        cl_event event;
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                               (const size_t *)&gWI,
                               (const size_t *)NULL, 0, 0, &event);
        events.push_back(event);
    }

    clWaitForEvents(events.size(), &events[0]);
    clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferBytes,
                        (void *)inputOutput, 0, NULL, NULL);
}

void checkProgramBuild(cl_int errNum, const cl_program& program,
                       const cl_device_id *deviceIDs)
{
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
}

int main(int argc, char **argv)
{
    cl_int errNum;
    cl_uint numPlatforms, numDevices;
    cl_platform_id *platformIDs;
    cl_device_id *deviceIDs;
    cl_context context;

    auto cmds = parseArgs(argc, argv);

    if (cmds.empty())
    {
        std::cerr << "No commands provided. Exiting..." << std::endl;
        return EXIT_FAILURE;
    }

    GET_PLATFORM_INFO(platformIDs, numPlatforms, errNum)
    GET_DEVICE_INFO(platformIDs, deviceIDs, numDevices, errNum);
    CREATE_CONTEXT(context, numDevices, deviceIDs, platformIDs, errNum)

    auto programStr = loadProgramStr();
    const char *src = programStr.c_str();
    size_t length = programStr.length();

    // Create program from source and build program
    cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    errNum = clBuildProgram(program, numDevices, deviceIDs, "-I.", NULL, NULL);
    checkProgramBuild(errNum, program, deviceIDs);

    std::array<int, BUFFER_SIZE> inputOutput{};
    std::iota(inputOutput.begin(), inputOutput.end(), 0);
    auto kernels = createKernels(cmds, program);
    funcTime(executeCommands, context, deviceIDs,
             kernels, numDevices, inputOutput.data());
#ifdef DEBUG
    print(numDevices, inputOutput.data());
#endif

    return 0;
}