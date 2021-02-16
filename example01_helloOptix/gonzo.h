void cudaFree(void* devPtr);
void cudaGetDeviceCount(int* count);
void optixInit();
#define OPTIX_CHECK(call)       call
