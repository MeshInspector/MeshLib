#define MRC_BUILD_LIBRARY
#include "MRCCuda/MRCudaBasic.h"

#include <cuda_placeholder.h>


bool MR_Cuda_isCudaAvailable(int *driverVersion, int *runtimeVersion, int *computeMajor, int *computeMinor)
{
    return ::MR::Cuda::isCudaAvailable(
        driverVersion,
        runtimeVersion,
        computeMajor,
        computeMinor
    );
}

