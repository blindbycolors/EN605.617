#include "HelperFunctions.cuh"

void buildGraph(float * h_weights,
                int * h_destinationOffsets,
                int * h_sourceIndices)
{
    h_weights [0] = 0.333333;
    h_weights [1] = 0.500000;
    h_weights [2] = 0.333333;
    h_weights [3] = 0.500000;
    h_weights [4] = 0.500000;
    h_weights [5] = 1.000000;
    h_weights [6] = 0.333333;
    h_weights [7] = 0.500000;
    h_weights [8] = 0.500000;
    h_weights [9] = 0.500000;

    h_destinationOffsets [0] = 0;
    h_destinationOffsets [1] = 1;
    h_destinationOffsets [2] = 3;
    h_destinationOffsets [3] = 4;
    h_destinationOffsets [4] = 6;
    h_destinationOffsets [5] = 8;
    h_destinationOffsets [6] = 10;

    h_sourceIndices [0] = 2;
    h_sourceIndices [1] = 0;
    h_sourceIndices [2] = 2;
    h_sourceIndices [3] = 0;
    h_sourceIndices [4] = 4;
    h_sourceIndices [5] = 5;
    h_sourceIndices [6] = 2;
    h_sourceIndices [7] = 3;
    h_sourceIndices [8] = 3;
    h_sourceIndices [9] = 4;
}

void initializeGraph(nvgraphHandle_t& handle,
                     nvgraphGraphDescr_t &graph,
                     int * h_destinationOffsets,
                     int * h_sourceIndices,
                     float * h_weights)
{
    cudaDataType_t edgeDimT = CUDA_R_32F;
    cudaDataType_t* vertexDimT;
    nvgraphCSCTopology32I_t CSCInput;

    vertexDimT =
            (cudaDataType_t*)malloc(VERTEX_NUM_SETS * sizeof(cudaDataType_t));
    CSCInput =
            (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));

    vertexDimT[0] = CUDA_R_32F;
    vertexDimT[1]= CUDA_R_32F;

    CSCInput->nvertices = NUM_VERTEX;
    CSCInput->nedges = NUM_EDGES;
    CSCInput->destination_offsets = (int *) h_destinationOffsets;
    CSCInput->source_indices = (int *) h_sourceIndices;

    // Set graph connectivity and properties (tranfers)
    checkNvGraph(nvgraphSetGraphStructure(handle, graph, (void *) CSCInput,
                                          NVGRAPH_CSC_32));
    checkNvGraph(nvgraphAllocateVertexData(handle, graph, VERTEX_NUM_SETS,
                                           vertexDimT));
    checkNvGraph(
            nvgraphAllocateEdgeData(handle, graph, EDGE_NUM_SETS, &edgeDimT));
    checkNvGraph(nvgraphSetEdgeData(handle, graph, (void *) h_weights, 0));

    freeMalloc(vertexDimT, CSCInput);
}

void runNvGraphWidestPath(int * h_destinationOffsets,
                          int * h_sourceIndices,
                          float * h_weights,
                          float * h_widestPath1,
                          float * h_widestPath2,
                          void** vertexDim)
{
    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;

    vertexDim[0]= (void*) h_widestPath1;
    vertexDim[1]= (void*) h_widestPath2;

    checkNvGraph(nvgraphCreate(&handle));
    checkNvGraph(nvgraphCreateGraphDescr(handle, &graph));
    initializeGraph(handle, graph, h_destinationOffsets, h_sourceIndices,
                    h_weights);

    // Solve for widest path with source 0
    int source_vert = 0;
    checkNvGraph(nvgraphWidestPath(handle, graph, 0, &source_vert, 0));

    // Solve for widest path with source 2
    source_vert = 2;
    checkNvGraph(nvgraphWidestPath(handle, graph, 0, &source_vert, 1));

    // Get and print result
    checkNvGraph(nvgraphGetVertexData(handle, graph, (void *) h_widestPath1, 0));
    checkNvGraph(nvgraphGetVertexData(handle, graph, (void *) h_widestPath2, 1));

    //Clean
    checkNvGraph(nvgraphDestroyGraphDescr(handle, graph));
    checkNvGraph(nvgraphDestroy(handle));
}

void printResults(float * widestPathResults)
{
    for (auto i = 0; i < NUM_VERTEX; i++)
    {
        printf("%f\n", widestPathResults[i]);
    }
    printf("\n");
    printf("\nDone!\n");
}
