#include <cstdlib>
void cudaFree(void* devPtr);
void cudaGetDeviceCount(int* count);
void optixInit();
#define OPTIX_CHECK(call)       call
//typedef unsigned int CUdeviceptr;
typedef void* CUdeviceptr;
void cudaMalloc(void** ptr, size_t size);
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
struct OptixProgramGroup_t {};
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
