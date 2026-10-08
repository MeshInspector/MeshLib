#pragma once

#include "MRMeshFwd.h"

#include <utility>

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
    RGB8, ///< 8-bit red, green and blue components; the scalar value is their luma
    RGBA8, ///< 8-bit red, green, blue and alpha components as in Color; the scalar value is the luma of the color components
    Unknown,
    Count
};

/// returns the size in bytes of a value of given type, or 0 for ScalarType::Unknown
[[nodiscard]] MRMESH_API size_t getScalarTypeSize( ScalarType scalarType );

/// returns the minimal and the maximal values of given integer type (of a color component for RGB8 and RGBA8),
/// or zeros for floating-point types and ScalarType::Unknown
[[nodiscard]] MRMESH_API std::pair<std::int64_t, std::uint64_t> getScalarTypeMinMax( ScalarType scalarType );

/// get a function to convert binary data of specified format type to a scalar value;
/// floating-point values are returned as they are
/// \param scalarType - binary format type
/// \param range - (for integer and color types only) the range of possible values
/// \param min - (for integer and color types only) the minimal value
MRMESH_API std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, std::uint64_t range, std::int64_t min );


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
        case ScalarType::RGB8:
        case ScalarType::RGBA8:
        {
            // the same luma as in convertImageToDistanceMap
            const auto* rgb = ( const uint8_t* )c;
            return f( 0.299f * float( rgb[0] ) + 0.587f * float( rgb[1] ) + 0.114f * float( rgb[2] ) );
        }
        case ScalarType::Unknown:
            return {};
        case ScalarType::Count:
            MR_UNREACHABLE
    }
    MR_UNREACHABLE
#undef M
}


} // namespace MR
