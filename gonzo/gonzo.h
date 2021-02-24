#pragma once
#include <cstdlib>
void cudaGetDeviceCount(int* count);
void optixInit();
#define OPTIX_CHECK(call)       call
//typedef unsigned int CUdeviceptr;
typedef void* CUdeviceptr;
enum cudaError {
        cudaSuccess = 0,
        cudaErrorUnknown = 999
};
typedef enum cudaError cudaError_t;
cudaError_t cudaMalloc(void** ptr, size_t size);
cudaError_t cudaFree(void* devPtr);
#define CUDA_CHECK(call)        cuda##call
enum cudaMemcpyKind {
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2
};
void cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
struct CUctx_st {};
typedef CUctx_st* CUcontext;
struct CUstream_st {};
typedef CUstream_st* CUstream;
struct cudaDeviceProp {
        char name[256];
};
typedef struct OptixDeviceContext_t* OptixDeviceContext;
typedef struct OptixPipeline_t* OptixPipeline;
struct OptixPipelineCompileOptions {
        int usesMotionBlur;
        unsigned int traversableGraphFlags;
        int numPayloadValues;
        int numAttributeValues;
        unsigned int exceptionFlags;
        const char * pipelineLaunchParamsVariableName;
};
enum OptixExceptionFlags {
        OPTIX_EXCEPTION_FLAG_NONE
};
enum OptixTraversableGraphFlags {
  OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0,
  OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1u << 0,
  OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1u << 1
};
struct OptixPipelineLinkOptions {
        unsigned int maxTraceDepth;
};
struct OptixModule_t {};
typedef OptixModule_t OptixModule;
enum OptixCompileOptimizationLevel {
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT
};
enum OptixCompileDebugLevel {
        OPTIX_COMPILE_DEBUG_LEVEL_NONE
};
struct OptixModuleCompileOptions {
        int maxRegisterCount;
        OptixCompileOptimizationLevel 	optLevel;
        OptixCompileDebugLevel debugLevel;
};
struct OptixProgramGroup_t {
        const char* functionName;
};
typedef OptixProgramGroup_t OptixProgramGroup;
struct OptixShaderBindingTable {
        CUdeviceptr raygenRecord;
        CUdeviceptr missRecordBase;
        unsigned int missRecordStrideInBytes;
        unsigned int missRecordCount;
        CUdeviceptr hitgroupRecordBase;
        unsigned int hitgroupRecordStrideInBytes;
        unsigned int hitgroupRecordCount;
};

#define __align__(alignment)
#define __forceinline__
#define __device__
#define OPTIX_SBT_RECORD_HEADER_SIZE   ( (size_t)32 )

void cudaSetDevice(int device);
typedef CUstream_st* cudaStream_t;
void cudaStreamCreate(cudaStream_t* pStream);
void cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
enum CUresult {
        CUDA_SUCCESS = 0
};
CUresult cuCtxGetCurrent(CUcontext* pctx);
enum OptixResult {
        OPTIX_SUCCESS
};
struct OptixDeviceContextOptions {};
OptixResult optixDeviceContextCreate(
                CUcontext fromContext,
                const OptixDeviceContextOptions* options,
                OptixDeviceContext* context);
typedef void( * OptixLogCallback)(unsigned int level, const char *tag, const char *message, void *cbdata);
OptixResult optixDeviceContextSetLogCallback(
                OptixDeviceContext context,
                OptixLogCallback callbackFunction,
                void *callbackData,
                unsigned int callbackLevel);

OptixResult optixModuleCreateFromPTX(
                OptixDeviceContext context,
                const OptixModuleCompileOptions *moduleCompileOptions,
                const OptixPipelineCompileOptions *pipelineCompileOptions,
                const char *PTX,
                size_t PTXsize,
                char *logString,
                size_t *logStringSize,
                OptixModule *module);
struct OptixProgramGroupOptions {};
enum OptixProgramGroupKind {
        OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        OPTIX_PROGRAM_GROUP_KIND_MISS,
        OPTIX_PROGRAM_GROUP_KIND_HITGROUP
};
struct OptixProgramGroupSingleModule {
        OptixModule module;
        const char * entryFunctionName;
};
struct OptixProgramGroupCallables {};
struct OptixProgramGroupHitgroup {
        OptixModule moduleCH;
        const char * entryFunctionNameCH;
        OptixModule moduleAH;
        const char * entryFunctionNameAH;
        OptixModule moduleIS;
        const char * entryFunctionNameIS;
};
struct OptixProgramGroupDesc {
        OptixProgramGroupKind kind;
        union {
                OptixProgramGroupSingleModule   raygen;
                OptixProgramGroupSingleModule   miss;
                OptixProgramGroupSingleModule   exception;
                OptixProgramGroupCallables   callables;
                OptixProgramGroupHitgroup   hitgroup;
        };
};
OptixResult optixProgramGroupCreate(
        OptixDeviceContext context,
        const OptixProgramGroupDesc *programDescriptions,
        unsigned int numProgramGroups,
        const OptixProgramGroupOptions *options,
        char *logString,
        size_t *logStringSize,
        OptixProgramGroup *programGroups);
OptixResult optixPipelineCreate(
                OptixDeviceContext context,
		const OptixPipelineCompileOptions * pipelineCompileOptions,
		const OptixPipelineLinkOptions * pipelineLinkOptions,
		const OptixProgramGroup * programGroups,
		unsigned int numProgramGroups,
		char * logString,
		size_t * logStringSize,
		OptixPipeline * pipeline);
OptixResult optixPipelineSetStackSize(
                OptixPipeline  	pipeline,
		unsigned int  	directCallableStackSizeFromTraversal,
		unsigned int  	directCallableStackSizeFromState,
		unsigned int  	continuationStackSize,
		unsigned int  	maxTraversableGraphDepth);
OptixResult optixSbtRecordPackHeader(
                OptixProgramGroup programGroup,
		void * sbtRecordHeaderHostPointer);
OptixResult optixLaunch(
                OptixPipeline  	pipeline,
		CUstream  	stream,
		CUdeviceptr  	pipelineParams,
		size_t  	pipelineParamsSize,
		const OptixShaderBindingTable *  	sbt,
		unsigned int  	width,
		unsigned int  	height,
		unsigned int  	depth
	);
#define CUDA_SYNC_CHECK()
typedef unsigned long long OptixTraversableHandle;
enum OptixBuildInputType {
        OPTIX_BUILD_INPUT_TYPE_TRIANGLES
};
enum OptixVertexFormat {
        OPTIX_VERTEX_FORMAT_FLOAT3
};
enum OptixIndicesFormat {
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3
};
enum OptixTransformFormat {};
struct OptixBuildInputTriangleArray {
        const CUdeviceptr* vertexBuffers;
        unsigned int numVertices;
        OptixVertexFormat vertexFormat;
        unsigned int vertexStrideInBytes;
        CUdeviceptr indexBuffer;
        unsigned int numIndexTriplets;
        OptixIndicesFormat indexFormat;
        unsigned int indexStrideInBytes;
        CUdeviceptr preTransform;
        const unsigned int* flags;
        unsigned int numSbtRecords;
        CUdeviceptr sbtIndexOffsetBuffer;
        unsigned int sbtIndexOffsetSizeInBytes;
        unsigned int sbtIndexOffsetStrideInBytes;
        unsigned int primitiveIndexOffset;
        OptixTransformFormat transformFormat;
};
struct OptixBuildInput {
        OptixBuildInputType type;
        union {
                OptixBuildInputTriangleArray triangleArray;
        };
};
enum OptixBuildFlags {
        OPTIX_BUILD_FLAG_NONE,
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION
};
struct OptixMotionOptions {
        unsigned short numKeys;
        unsigned short flags;
        float timeBegin;
        float timeEnd;
};
enum OptixBuildOperation {
        OPTIX_BUILD_OPERATION_BUILD
};
struct OptixAccelBuildOptions {
        unsigned int buildFlags;
        OptixBuildOperation operation;
        OptixMotionOptions motionOptions;
};
struct OptixAccelBufferSizes {
        size_t outputSizeInBytes;
        size_t tempSizeInBytes;
        size_t tempUpdateSizeInBytes;
};
OptixResult optixAccelComputeMemoryUsage(
                OptixDeviceContext context,
		const OptixAccelBuildOptions *accelOptions,
		const OptixBuildInput *buildInputs,
		unsigned int numBuildInputs,
		OptixAccelBufferSizes * bufferSizes);
enum OptixAccelPropertyType {
        OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
};
struct OptixAccelEmitDesc {
        CUdeviceptr result;
        OptixAccelPropertyType type;
};
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
	unsigned int  	numEmittedProperties);
OptixResult optixAccelCompact(
        OptixDeviceContext  	context,
	CUstream  	stream,
	OptixTraversableHandle  	inputHandle,
	CUdeviceptr  	outputBuffer,
	size_t  	outputBufferSizeInBytes,
	OptixTraversableHandle *  	outputHandle);

extern "C" {
unsigned int optixGetPayload_0();
unsigned int optixGetPayload_1();
unsigned int optixGetPrimitiveIndex();
}
typedef unsigned int OptixVisibilityMask;
enum OptixRayFlags {
        OPTIX_RAY_FLAG_DISABLE_ANYHIT = 1u << 0
};

#include "../common/gdt/gdt/math/vec.h"
typedef const gdt::vec_t<float, 3> float3;

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
	unsigned int &  	p1);
}
