#pragma once

#include "MRMesh/MRExpected.h"

namespace MR::Cuda
{

/// Process data by blocks alternately by GPU and CPU. GPUFunc is executed in a separate thread.
/// Both GPUFunc and CPUFunc must satisfy the following signature:
/// MR::Expected<void> (*funcName) ( BufferType& buffer, InputIt::value_type value )
template <typename BufferType, typename InputIt, typename GPUFunc, typename CPUFunc>
Expected<void> cudaPipeline( BufferType init, InputIt begin, InputIt end, GPUFunc gpuFunc, CPUFunc cpuFunc );

} // namespace MR::Cuda

#include "MRCudaPipeline.hpp"
