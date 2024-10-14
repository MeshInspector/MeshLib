#include "MRMakeSphereMesh.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMakeSphereMesh.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( SphereParams )

static_assert( sizeof( MRSphereParams ) == sizeof( SphereParams ) );

MRSphereParams mrSphereParamsNew( void )
{
    static const SphereParams result;
    RETURN( result );
}

MRMesh* mrMakeSphere( const MRSphereParams* params_ )
{
    ARG( params );
    RETURN_NEW( makeSphere( params ) );
}

MRMakeUVSphereParameters mrMakeUvSphereParametersNew( void )
{
    return {
        .radius = 1.0,
        .horizontalResolution = 16,
        .verticalResolution = 16,
    };
}

MRMesh* mrMakeUVSphere( const MRMakeUVSphereParameters* params )
{
    RETURN_NEW( makeUVSphere(
        params->radius,
        params->horizontalResolution,
        params->verticalResolution
    ) );
}
