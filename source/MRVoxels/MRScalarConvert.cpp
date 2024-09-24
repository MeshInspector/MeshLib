#include "MRScalarConvert.h"

namespace MR
{

std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, uint64_t range, int64_t min )
{
    switch ( scalarType )
    {
        case ScalarType::UInt8:
            return [range, min]( const char* c )
            {
                return float( *(const uint8_t*)(c) -min ) / float( range );
            };
        case ScalarType::UInt16:
            return [range, min]( const char* c )
            {
                return float( *(const uint16_t*)(c) -min ) / float( range );
            };
        case ScalarType::Int8:
            return [range, min]( const char* c )
            {
                return float( *(const int8_t*)(c) -min ) / float( range );
            };
        case ScalarType::Int16:
            return [range, min]( const char* c )
            {
                return float( *(const int16_t*)(c) -min ) / float( range );
            };
        case ScalarType::Int32:
            return [range, min]( const char* c )
            {
                return float( *(const int32_t*)(c) -min ) / float( range );
            };
        case ScalarType::UInt32:
            return [range, min]( const char* c )
            {
                return float( *(const uint32_t*)(c) -min ) / float( range );
            };
        case ScalarType::UInt64:
            return [range, min]( const char* c )
            {
                return float( *(const uint64_t*)(c) -min ) / float( range );
            };
        case ScalarType::Int64:
            return [range, min]( const char* c )
            {
                return float( *(const int64_t*)(c) -min ) / float( range );
            };
        case ScalarType::Float32:
            return []( const char* c )
            {
                return *(const float*)( c );
            };
        case ScalarType::Float64:
            return []( const char* c )
            {
                return float( *(const double*)( c ) );
            };
        case ScalarType::Float32_4:
            return []( const char* c )
            {
                return ( (const float*)c )[3];
            };
        case ScalarType::Unknown:
            return {};
        case ScalarType::Count:
            MR_UNREACHABLE
    }
    MR_UNREACHABLE
}

} // namespace MR
