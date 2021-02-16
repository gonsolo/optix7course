//  gonzo

//#include <stdlib.h>

#define __constant__
#define __global__

struct launchIndex {
        int x;
        int y;
};

launchIndex optixGetLaunchIndex();

//#define __device__

//void* malloc(size_t size) { return NULL; }

//#define __CLANG_CUDA_WRAPPERS_NEW

//namespace std {
//enum class align_val_t : std::size_t {};
//struct nothrow_t {};
//extern const std::nothrow_t nothrow;

//}

