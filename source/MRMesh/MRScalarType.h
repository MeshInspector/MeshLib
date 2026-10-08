#pragma once

#include "MRMeshFwd.h"

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

} // namespace MR
