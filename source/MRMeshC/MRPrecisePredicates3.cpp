#include "MRPrecisePredicates3.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRPrecisePredicates3.h"

using namespace MR;

REGISTER_AUTO_CAST( ConvertToIntVector )
REGISTER_AUTO_CAST( ConvertToFloatVector )

void mrConvertToIntVectorFree( MRConvertToIntVector* conv_ )
{
    ARG_PTR( conv );
    delete conv;
}

void mrConvertToFloatVectorFree( MRConvertToFloatVector* conv_ )
{
    ARG_PTR( conv );
    delete conv;
}
