#pragma once

#include "MRVoxelsFwd.h"

namespace MR
{

/// scalar value's binary format type
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

/// get a function to convert binary data of specified format type to a scalar value
/// \param scalarType - binary format type
/// \param range - (for integer types only) the range of possible values
/// \param min - (for integer types only) the minimal value
MRVOXELS_API std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, uint64_t range, int64_t min );

} // namespace MR
