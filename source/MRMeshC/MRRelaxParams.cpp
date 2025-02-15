#include "MRRelaxParams.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRRelaxParams.h"

using namespace MR;

REGISTER_AUTO_CAST( RelaxParams )

static_assert( sizeof( RelaxParams ) == sizeof( MRRelaxParams ) );

MRRelaxParams mrRelaxParamsNew()
{
    static const RelaxParams def {};
    RETURN( def );
}
