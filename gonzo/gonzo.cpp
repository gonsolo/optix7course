#include "gonzo.h"
#include "optix_device.h"
#include "dlfcn.h"
#include <cstdlib>
#include <fstream>
#include <iostream>

void cudaStreamCreate(cudaStream_t* pStream) {}
void cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {}
void cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {}
void cudaSetDevice(int device) {}

void cudaGetDeviceCount(int* count) {
        *count = 1;
}

cudaError_t cudaMalloc(void** ptr, size_t size) {
        *ptr = malloc(size);
        return *ptr? cudaSuccess : cudaErrorUnknown;
}

cudaError_t cudaFree(void* devPtr) {
        if (devPtr) free(devPtr);
        return cudaSuccess;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) { return CUDA_SUCCESS; }

void optixInit() {}
OptixResult optixDeviceContextCreate(
        CUcontext fromContext,
        const OptixDeviceContextOptions* options,
        OptixDeviceContext* context) { return OPTIX_SUCCESS; }
OptixResult optixDeviceContextSetLogCallback(
        OptixDeviceContext context,
        OptixLogCallback callbackFunction,
        void *callbackData,
        unsigned int callbackLevel) { return OPTIX_SUCCESS; }
OptixResult optixModuleCreateFromPTX(
        OptixDeviceContext context,
        const OptixModuleCompileOptions *moduleCompileOptions,
        const OptixPipelineCompileOptions *pipelineCompileOptions,
        const char *PTX,
        size_t PTXsize,
        char *logString,
        size_t *logStringSize,
        OptixModule *module) {
        
        std::ofstream dummy("dummy.cpp");
        dummy << "int " << pipelineCompileOptions->pipelineLaunchParamsVariableName << ";" << std::endl;
        dummy.close();

        return OPTIX_SUCCESS;
}

OptixResult optixProgramGroupCreate(
        OptixDeviceContext context,
        const OptixProgramGroupDesc *programDescriptions,
        unsigned int numProgramGroups,
        const OptixProgramGroupOptions *options,
        char *logString,
        size_t *logStringSize,
        OptixProgramGroup *programGroups) {

        if (programDescriptions->kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN) {
                programGroups->functionName = programDescriptions->raygen.entryFunctionName;
        }
        return OPTIX_SUCCESS;
}

void (*raygen)();

OptixResult optixPipelineSetStackSize(
        OptixPipeline  	pipeline,
	unsigned int  	directCallableStackSizeFromTraversal,
	unsigned int  	directCallableStackSizeFromState,
	unsigned int  	continuationStackSize,
	unsigned int  	maxTraversableGraphDepth) { return OPTIX_SUCCESS; }
OptixResult optixSbtRecordPackHeader(
        OptixProgramGroup programGroup,
        void* sbtRecordHeaderHostPointer) { return OPTIX_SUCCESS; }
OptixResult optixLaunch(
        OptixPipeline  	pipeline,
        CUstream  	stream,
        CUdeviceptr  	pipelineParams,
        size_t  	pipelineParamsSize,
        const OptixShaderBindingTable *  	sbt,
        unsigned int  	width,
        unsigned int  	height,
        unsigned int  	depth) {

        std::cout << "gonzo launch: " << width << " " << height << std::endl;

        raygen();

        return OPTIX_SUCCESS;
}


OptixResult optixPipelineCreate(
                OptixDeviceContext context,
		const OptixPipelineCompileOptions * pipelineCompileOptions,
		const OptixPipelineLinkOptions * pipelineLinkOptions,
		const OptixProgramGroup * programGroups,
		unsigned int numProgramGroups,
		char * logString,
		size_t * logStringSize,
		OptixPipeline * pipeline) {

        std::system("clang++ -fpic -c dummy.cpp");
        std::system("clang++ -fpic -xc++ -std=c++17 -I ../gonzo/ -I ../common/gdt -I ../gonzo/optix_device.h  -c devicePrograms.cu");
        std::system("clang++ -fpic -shared -o dummy.so dummy.o devicePrograms.o");

        void* handle = dlopen("./dummy.so", RTLD_LAZY);
        if (!handle) {
                std::cerr << "dlopen failed!" << std::endl;
                exit(EXIT_FAILURE);
        }
        *(void**)(&raygen) = dlsym(handle, "__raygen__renderFrame");

        return OPTIX_SUCCESS;
}

extern "C" {
        char embedded_ptx_code[256];
}

uint3 optixGetLaunchIndex() {
        return uint3();
}

