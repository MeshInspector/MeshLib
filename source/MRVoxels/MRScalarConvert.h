#pragma once

#include "MRVoxelsFwd.h"

namespace MR
{

/// ...
enum class ScalarType
{
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
    Float32_4, ///< the last value from float[4]
    Unknown,
    Count
};

/// ...
MRVOXELS_API std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, uint64_t range, int64_t min );

} // namespace MR
