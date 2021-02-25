#include "gonzo.h"
#include "optix_device.h"
#include "dlfcn.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

extern "C" {
void accelBuild(
        unsigned int numIndices,
        uint32_t* indices,
        unsigned int numVertices,
        float* vertices);

void trace(float ox, float oy, float oz,
           float dx, float dy, float dz,
           float tmax,
           int* result);
}

void cudaStreamCreate(cudaStream_t* pStream) {}
void cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {}
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

void cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
        memcpy(dst, src, count);
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
void (*closest)();
void (*miss)();

OptixResult optixPipelineSetStackSize(
        OptixPipeline  	pipeline,
	unsigned int  	directCallableStackSizeFromTraversal,
	unsigned int  	directCallableStackSizeFromState,
	unsigned int  	continuationStackSize,
	unsigned int  	maxTraversableGraphDepth) { return OPTIX_SUCCESS; }
OptixResult optixSbtRecordPackHeader(
        OptixProgramGroup programGroup,
        void* sbtRecordHeaderHostPointer) { return OPTIX_SUCCESS; }

void* launch;

int x;
int y;

const OptixShaderBindingTable* shaderBindingTable;

OptixResult optixLaunch(
        OptixPipeline  	pipeline,
        CUstream  	stream,
        CUdeviceptr  	pipelineParams,
        size_t  	pipelineParamsSize,
        const OptixShaderBindingTable *  	sbt,
        unsigned int  	width,
        unsigned int  	height,
        unsigned int  	depth) {

        memcpy(launch, pipelineParams, pipelineParamsSize);
        shaderBindingTable = sbt;
        for(y = 0; y < height; y++) {
                for(x = 0; x < width; x++) {
                        raygen();
                }
        }

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

        std::ofstream dummy("dummy.cpp");
        dummy << "char " << pipelineCompileOptions->pipelineLaunchParamsVariableName << "[256];" << std::endl;
        dummy.close();
        std::system("clang++ -g -fpic -c dummy.cpp");
        std::system("clang -g -fpic -xc++ -I ../gonzo/ -I ../common/gdt -I ../gonzo/optix_device.h  -c devicePrograms.cu");
        std::system("clang++ -g -fpic -shared -o dummy.so dummy.o devicePrograms.o");

        void* handle = dlopen("./dummy.so", RTLD_LAZY);
        if (!handle) {
                std::cerr << "dlopen failed!" << std::endl;
                exit(EXIT_FAILURE);
        }
        *(void**)(&raygen) = dlsym(handle, "__raygen__renderFrame");
        if (!raygen) {
                std::cerr << "no raygen!" << std::endl;
                exit(EXIT_FAILURE);
        }
        *(void**)(&closest) = dlsym(handle, "__closesthit__radiance");
        if (!closest) {
                std::cerr << "no closest!" << std::endl;
                exit(EXIT_FAILURE);
        }
        *(void**)(&miss) = dlsym(handle, "__miss__radiance");
        if (!miss) {
                std::cerr << "no miss!" << std::endl;
                exit(EXIT_FAILURE);
        }
        launch = dlsym(handle, pipelineCompileOptions->pipelineLaunchParamsVariableName);
        return OPTIX_SUCCESS;
}

extern "C" {
        char embedded_ptx_code[256];
}

uint3 optixGetLaunchIndex() {
        uint3 index;
        index.x = x;
        index.y = y;
        index.z = 0;
        return index;
}

OptixResult optixAccelComputeMemoryUsage(
        OptixDeviceContext context,
        const OptixAccelBuildOptions *accelOptions,
        const OptixBuildInput *buildInputs,
        unsigned int numBuildInputs,
        OptixAccelBufferSizes * bufferSizes) { return OPTIX_SUCCESS; }

OptixResult optixAccelBuild(
        OptixDeviceContext context,
	CUstream stream,
	const OptixAccelBuildOptions * accelOptions,
	const OptixBuildInput * buildInputs,
	unsigned int  	numBuildInputs,
	CUdeviceptr  	tempBuffer,
	size_t  	tempBufferSizeInBytes,
	CUdeviceptr  	outputBuffer,
	size_t  	outputBufferSizeInBytes,
	OptixTraversableHandle * outputHandle,
	const OptixAccelEmitDesc * emittedProperties,
	unsigned int  	numEmittedProperties) {

        auto& triangleArray = buildInputs->triangleArray;
        auto indices = (uint32_t*)triangleArray.indexBuffer;
        auto numIndices = triangleArray.numIndexTriplets * 3;
        auto numVertices = triangleArray.numVertices;
        auto vertices = (float*)triangleArray.vertexBuffers[0];

        accelBuild(numIndices, indices, numVertices, vertices);

        return OPTIX_SUCCESS;
}

OptixResult optixAccelCompact(
        OptixDeviceContext  	context,
	CUstream  	stream,
	OptixTraversableHandle  	inputHandle,
	CUdeviceptr  	outputBuffer,
	size_t  	outputBufferSizeInBytes,
	OptixTraversableHandle *  	outputHandle) { return OPTIX_SUCCESS; }

static void *unpackPointer(uint32_t i0, uint32_t i1) {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void* ptr = reinterpret_cast<void*>( uptr );
        return ptr;
}


int counter = 0;
int primitiveIndex = 0;
unsigned int payload0 = 0;
unsigned int payload1 = 0;
float rayDirX;
float rayDirY;
float rayDirZ;

extern "C" {

void optixTrace(
        OptixTraversableHandle  	handle,
	float3  	rayOrigin,
	float3  	rayDirection,
	float  	tmin,
	float  	tmax,
	float  	rayTime,
	OptixVisibilityMask  	visibilityMask,
	unsigned int  	rayFlags,
	unsigned int  	SBToffset,
	unsigned int  	SBTstride,
	unsigned int  	missSBTIndex,
	unsigned int &  	p0,
	unsigned int &  	p1) {

        int* pointer = (int*)unpackPointer(p0, p1);
        rayDirX = rayDirection.x;
        rayDirY = rayDirection.y;
        rayDirZ = rayDirection.z;

        trace(rayOrigin.x, rayOrigin.y, rayOrigin.z,
              rayDirection.x, rayDirection.y, rayDirection.z,
              tmax,
              pointer);

        payload0 = p0;
        payload1 = p1;

        if (*pointer == 0) {
                miss();                
        } else {
                primitiveIndex = *pointer;
                closest();
        }
}

unsigned int optixGetPrimitiveIndex() {
        return primitiveIndex;
}

unsigned int optixGetPayload_0() {
        return payload0;
}

unsigned int optixGetPayload_1() {
        return payload1;
}

CUdeviceptr optixGetSbtDataPointer() {
        char* base = (char*)shaderBindingTable->hitgroupRecordBase;
        CUdeviceptr data = base + OPTIX_SBT_RECORD_HEADER_SIZE;
        return data; 
}

} // extern "C"

float3 optixGetWorldRayDirection() {
        float3 direction;
        direction.x = rayDirX;        
        direction.y = rayDirY;        
        direction.z = rayDirZ;        
        return direction;
}

