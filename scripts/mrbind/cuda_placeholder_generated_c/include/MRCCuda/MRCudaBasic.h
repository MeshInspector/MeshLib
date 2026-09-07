#pragma once

#include <MRCMisc/exports.h>

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


// Returns true if Cuda is present on this GPU.
// Since Cuda is not supported on this platform, this function always returns false.
/// Generated from function `MR::Cuda::isCudaAvailable`.
/// Parameter `driverVersion` defaults to a null pointer in C++.
/// Parameter `runtimeVersion` defaults to a null pointer in C++.
/// Parameter `computeMajor` defaults to a null pointer in C++.
/// Parameter `computeMinor` defaults to a null pointer in C++.
MRC_API bool MR_Cuda_isCudaAvailable(int *driverVersion, int *runtimeVersion, int *computeMajor, int *computeMinor);

#ifdef __cplusplus
} // extern "C"
#endif
