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
MRVOXELS_API std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, Uint64 range, Int64 min );


/// More general template to pass a single value of specified format \p scalarType to a generic function \p f
template <typename F>
std::invoke_result_t<F, int> visitScalarType( F&& f, ScalarType scalarType, const char* c )
{
#define M(T) return f( *( const T* )( c ) );

    switch ( scalarType )
    {
        case ScalarType::UInt8:
            M( uint8_t )
        case ScalarType::UInt16:
            M( uint16_t )
        case ScalarType::Int8:
            M( int8_t )
        case ScalarType::Int16:
            M( int16_t )
        case ScalarType::Int32:
            M( int32_t )
        case ScalarType::UInt32:
            M( uint32_t )
        case ScalarType::UInt64:
            M( uint64_t )
        case ScalarType::Int64:
            M( int64_t )
        case ScalarType::Float32:
            M( float )
        case ScalarType::Float64:
            M( double )
        case ScalarType::Float32_4:
            return f( *((const float*)c + 3 ) );
        case ScalarType::Unknown:
            return {};
        case ScalarType::Count:
            MR_UNREACHABLE
    }
    MR_UNREACHABLE
#undef M
}


} // namespace MR
