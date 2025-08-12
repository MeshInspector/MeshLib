// This file provides a stub

namespace MR::Cuda
{

// Returns true if Cuda is present on this GPU.
// Since Cuda is not supported on this platform, this function always returns false.
inline bool isCudaAvailable( int* driverVersion = nullptr, int* runtimeVersion = nullptr, int* computeMajor = nullptr, int* computeMinor = nullptr )
{
    (void)driverVersion;
    (void)runtimeVersion;
    (void)computeMajor;
    (void)computeMinor;
    return false;
}

}
