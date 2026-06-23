// This file provides a stub

#include "MRMesh/MRExpected.h"

namespace MR::Cuda
{

struct RuntimeInfo
{
    /// maximum driver supported version
    int driverVersion = 0;

    /// current runtime version
    int runtimeVersion = 0;

    /// compute capability major version
    int computeMajor = 0;

    /// compute capability minor version
    int computeMinor = 0;

    /// returns true if all versions pass the checks
    [[nodiscard]] bool fitForComputations() const { return false; }
};

// Returns an error if CUDA is not available.
// Since Cuda is not supported on this platform, this function always returns an error.
inline Expected<RuntimeInfo> getRuntimeInfo()
{
    return unexpected( "CUDA is not supported on this platform" );
}

}
