#include "MRScalarConvert.h"

namespace MR
{

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

std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, std::uint64_t range, std::int64_t min )
{
    return [range, min, scalarType] ( const char* c )
    {
        return visitScalarType( [range, min] ( auto v ) { return float( v - min ) / float( range ); }, scalarType, c );
    };
}

} // namespace MR
