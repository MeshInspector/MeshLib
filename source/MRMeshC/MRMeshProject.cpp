#include "MRMeshProject.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshProject.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( MeshProjectionResult )
REGISTER_AUTO_CAST( Vector3f )

namespace
{

MeshPart cast( MRMeshPart mp )
{
    return {
        *auto_cast( mp.mesh ),
        auto_cast( mp.region )
    };
}

} // namespace

static_assert( sizeof( MRMeshProjectionResult ) == sizeof( MeshProjectionResult ) );

MRFindProjectionParameters mrFindProjectionParametersNew( void )
{
    return {
        .upDistLimitSq = FLT_MAX,
        .xf = nullptr,
        .loDistLimitSq = 0,
    };
}

MRMeshProjectionResult mrFindProjection( const MRVector3f* pt_, const MRMeshPart* mp, const MRFindProjectionParameters* params )
{
    ARG( pt );
    RETURN( findProjection( pt, cast( *mp ), params->upDistLimitSq, auto_cast( params->xf ), params->loDistLimitSq ) );
}
