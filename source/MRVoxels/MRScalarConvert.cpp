#include "MRScalarConvert.h"

namespace MR
{

std::function<float ( const char* )> getTypeConverter( ScalarType scalarType, std::uint64_t range, std::int64_t min )
{
    return [range, min, scalarType] ( const char* c )
    {
        return visitScalarType( [range, min] ( auto v ) { return float( v - min ) / float( range ); }, scalarType, c );
    };
}

} // namespace MR
