#include "MRScalarConvert.h"

namespace MR
{

std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, Uint64 range, Int64 min )
{
    return [range, min, scalarType] ( const char* c )
    {
        return visitScalarType( [range, min] ( auto v ) { return float( v - min ) / float( range ); }, scalarType, c );
    };
}

} // namespace MR
