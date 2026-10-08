#include "MRScalarConvert.h"

#include <limits>

namespace MR
{

namespace
{

template <typename T>
MinMaxd minMaxOf()
{
    return { double( std::numeric_limits<T>::lowest() ), double( std::numeric_limits<T>::max() ) };
}

} // anonymous namespace

size_t getScalarTypeSize( ScalarType scalarType )
{
    switch ( scalarType )
    {
    case ScalarType::UInt8:
    case ScalarType::Int8:
        return 1;
    case ScalarType::UInt16:
    case ScalarType::Int16:
        return 2;
    case ScalarType::RGB8:
        return 3;
    case ScalarType::UInt32:
    case ScalarType::Int32:
    case ScalarType::Float32:
    case ScalarType::RGBA8:
        return 4;
    case ScalarType::UInt64:
    case ScalarType::Int64:
    case ScalarType::Float64:
        return 8;
    case ScalarType::Float32_4:
        return 16;
    case ScalarType::Unknown:
    case ScalarType::Count:
        break;
    }
    return 0;
}

MinMaxd getScalarTypeMinMax( ScalarType scalarType )
{
    switch ( scalarType )
    {
    case ScalarType::UInt8:
    case ScalarType::RGB8:
    case ScalarType::RGBA8:
        return minMaxOf<uint8_t>();
    case ScalarType::Int8:
        return minMaxOf<int8_t>();
    case ScalarType::UInt16:
        return minMaxOf<uint16_t>();
    case ScalarType::Int16:
        return minMaxOf<int16_t>();
    case ScalarType::UInt32:
        return minMaxOf<uint32_t>();
    case ScalarType::Int32:
        return minMaxOf<int32_t>();
    case ScalarType::UInt64:
        return minMaxOf<uint64_t>();
    case ScalarType::Int64:
        return minMaxOf<int64_t>();
    case ScalarType::Float32:
    case ScalarType::Float64:
    case ScalarType::Float32_4:
    case ScalarType::Unknown:
    case ScalarType::Count:
        break;
    }
    return { 0.0, 0.0 };
}

std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, double range, double min )
{
    if ( scalarType == ScalarType::Float32 || scalarType == ScalarType::Float64 || scalarType == ScalarType::Float32_4 )
    {
        return [scalarType] ( const char* c )
        {
            return visitScalarType( [] ( auto v ) { return float( v ); }, scalarType, c );
        };
    }
    return [range, min, scalarType] ( const char* c )
    {
        // the subtraction in double does not overflow for 64-bit types
        return visitScalarType( [range, min] ( auto v ) { return float( double( v ) - min ) / float( range ); }, scalarType, c );
    };
}

} // namespace MR
