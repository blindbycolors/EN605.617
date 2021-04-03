#include <iostream>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>
#include <string.h>
#include <chrono>
#include <helper_string.h>
#include <helper_cuda.h>
#include "HelperFunctions.cuh"


void nppImageProcessing(const std::string &nppFileName,
                        const std::string nppResultsFileName)
{
    npp::ImageCPU_8u_C1 oHostSrc;

    npp::loadImage(nppFileName, oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
    NppiSize kernelSize = {3, 1};
    NppiSize oSizeROI = {
            static_cast<int>(oHostSrc.width() - kernelSize.width + 1),
            static_cast<int>(oHostSrc.height() - kernelSize.height + 1)};
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    NppiPoint oAnchor = {2, 1};

    // Try get edge detection
    Npp32s hostKernel[3] = {-1, 0, 1};
    Npp32s *deviceKernel;
    size_t deviceKernelPitch;
    cudaMallocPitch((void **) &deviceKernel, &deviceKernelPitch,
                    kernelSize.width * sizeof(Npp32s),
                    kernelSize.height * sizeof(Npp32s));
    cudaMemcpy2D(deviceKernel, deviceKernelPitch, hostKernel,
                 sizeof(Npp32s) * kernelSize.width, // sPitch
                 sizeof(Npp32s) * kernelSize.width, // width
                 kernelSize.height, // height
                 cudaMemcpyHostToDevice);
    Npp32s divisor = 2;

    NPP_CHECK_NPP(nppiFilter_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                    oDeviceDst.data(), oDeviceDst.pitch(),
                                    oSizeROI, deviceKernel, kernelSize, oAnchor,
                                    divisor));
    NPP_CHECK_NPP (
            nppiAddC_8u_C1IRSfs(55, oDeviceDst.data(), oDeviceDst.pitch(),
                                oSizeROI, 0));

    // memcpy to host and clean up
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    saveImage(nppResultsFileName, oHostDst);
    std::cout << "Saved image: " << nppResultsFileName << std::endl;
    nppiFreeVars(oDeviceSrc.data(), oDeviceDst.data());
}

void runPageable()
{
    // Init host data
    ALLOCATE_PAGEABLE_MEMORY(h_destinationOffsets, int, NUM_VERTEX + 1);
    ALLOCATE_PAGEABLE_MEMORY(h_sourceIndices, int, NUM_EDGES);
    ALLOCATE_PAGEABLE_MEMORY(h_weights, float, NUM_EDGES);
    ALLOCATE_PAGEABLE_MEMORY(h_widestPath1, float, NUM_VERTEX);
    ALLOCATE_PAGEABLE_MEMORY(h_widestPath2, float, NUM_VERTEX);
    ALLOCATE_PAGEABLE_MEMORY(vertexDim, void*, VERTEX_NUM_SETS);

    auto startTime = std::chrono::high_resolution_clock::now();
    buildGraph(h_weights, h_destinationOffsets, h_sourceIndices);
    runNvGraphWidestPath(h_destinationOffsets,
                         h_sourceIndices,
                         h_weights,
                         h_widestPath1,
                         h_widestPath2,
                         vertexDim);

    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("\nPaged Memory Runtime: %f seconds\n",
           std::chrono::duration<double>(stopTime - startTime).count());

    printf("Widest Path from Source 0\n");
    printResults(h_widestPath1);

    printf("Widest Path from Source 2\n");
    printResults(h_widestPath2);

    freeMalloc(h_destinationOffsets, h_sourceIndices, h_weights,
               h_widestPath1, h_widestPath2, vertexDim);

    cudaDeviceReset();
}

void runPinned()
{
    // Init host data
    ALLOCATE_PINNED_MEMORY(h_destinationOffsets, int, (NUM_VERTEX + 1));
    ALLOCATE_PINNED_MEMORY(h_sourceIndices, int, NUM_EDGES);
    ALLOCATE_PINNED_MEMORY(h_weights, float, NUM_EDGES);
    ALLOCATE_PINNED_MEMORY(h_widestPath1, float, NUM_VERTEX);
    ALLOCATE_PINNED_MEMORY(h_widestPath2, float, NUM_VERTEX);
    ALLOCATE_PINNED_MEMORY(vertexDim, void*, VERTEX_NUM_SETS);

    auto startTime = std::chrono::high_resolution_clock::now();
    buildGraph(h_weights, h_destinationOffsets, h_sourceIndices);
    runNvGraphWidestPath(h_destinationOffsets,
                         h_sourceIndices,
                         h_weights,
                         h_widestPath1,
                         h_widestPath2,
                         vertexDim);

    auto stopTime = std::chrono::high_resolution_clock::now();
    printf("\nPinned Memory Runtime: %f seconds\n",
           std::chrono::duration<double>(stopTime - startTime).count());

    printf("Widest Path from Source 0\n");
    printResults(h_widestPath1);

    printf("Widest Path from Source 2\n");
    printResults(h_widestPath2);

    freePinned(h_destinationOffsets, h_sourceIndices, h_weights,
               h_widestPath1, h_widestPath2, vertexDim);
    //Clean
    cudaDeviceReset();
}

void nvgraphTraversal()
{
    runPageable();
    runPinned();
}

void processCmdLineArgs(int argc, char **argv, std::string& nppFileName)
{
    // Use command line specified CUDA device, otherwise use device with
    // highest Gflops/s
    int cuda_device = 0;
    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    if (deviceProp.major < 3)
    {
        printf("> nvGraph requires device SM 3.0+\n");
        printf("> Waiving.\n");
        exit(EXIT_WAIVED);
    }

    if(argc > 1)
    {
        for (auto i = 1; i < argc; ++i)
        {
            if(std::string("--image").compare(argv[1]))
            {
                nppFileName = argv[2];
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::string nppResultsFilename;
    std::string nppFileName = "mountains_gray.pgm";

    processCmdLineArgs(argc, argv, nppFileName);

    std::string::size_type dot = nppFileName.rfind('.');
    if (dot != std::string::npos)
    {
        nppResultsFilename = nppFileName.substr(0, dot);
        nppResultsFilename += "_filtered.pgm";
    }

    try
    {
      nppImageProcessing(nppFileName, nppResultsFilename);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        exit(-1);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        return -1;
    }

    nvgraphTraversal();
    return 0;
}
