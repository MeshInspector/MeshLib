#include "MRMeshCollide.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMeshCollide.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )

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

bool mrIsInside( const MRMeshPart * a, const MRMeshPart * b, const MRAffineXf3f * rigidB2A_ )
{
    ARG_PTR( rigidB2A );
    return isInside( cast( *a ), cast( *b ), rigidB2A );
}
