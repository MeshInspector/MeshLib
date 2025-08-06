#include "MRMeshMeshDistance.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshMeshDistance.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( MeshMeshDistanceResult )
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

static_assert( sizeof( MRMeshMeshDistanceResult ) == sizeof( MeshMeshDistanceResult ) );

MRMeshMeshDistanceResult mrFindDistance( const MRMeshPart* a, const MRMeshPart* b,
    const MRAffineXf3f* rigidB2A_, float upDistLimitSq )
{
    ARG_PTR( rigidB2A );
    RETURN( findDistance( cast( *a ), cast( *b ), rigidB2A, upDistLimitSq ) );
}
