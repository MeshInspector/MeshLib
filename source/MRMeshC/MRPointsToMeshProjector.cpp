#include "MRPointsToMeshProjector.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRPointsToMeshProjector.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_VECTOR_LIKE( MRScalars, float )

#define COPY_FROM( obj, field ) . field = auto_cast( ( obj ). field ) ,

MRMeshProjectionParameters mrMeshProjectionParametersNew( void )
{
    static const MeshProjectionParameters def {};
    return {
        COPY_FROM( def, loDistLimitSq )
        COPY_FROM( def, upDistLimitSq )
        .refXf = nullptr,
        .xf = nullptr,
    };
}

MRScalars* mrFindSignedDistances( const MRMesh* refMesh_, const MRMesh* mesh_, const MRMeshProjectionParameters* params_ )
{
    ARG( refMesh ); ARG( mesh );

    MeshProjectionParameters params;
    if ( params_ )
    {
        params = {
            COPY_FROM( *params_, loDistLimitSq )
            COPY_FROM( *params_, upDistLimitSq )
            .refXf = (AffineXf3f*)params_->refXf,
            .xf = (AffineXf3f*)params_->xf,
        };
    }

    RETURN_NEW_VECTOR( findSignedDistances( refMesh, mesh, params ) );
}
