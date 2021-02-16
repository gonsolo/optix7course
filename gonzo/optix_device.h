#include <cstdlib>

#define __constant__
#define __global__

struct uint3 {
        uint x;
        uint y;
        uint z;
};


uint3 optixGetLaunchIndex();

