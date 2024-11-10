#include "MRScalarConvert.h"

namespace MR
{

std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, uint64_t range, int64_t min )
{
    return [range, min, scalarType] ( const char* c )
    {
        return visitScalarType( [range, min] ( auto v ) { return float( v - min ) / float( range ); }, scalarType, c );
    };
}

} // namespace MR
