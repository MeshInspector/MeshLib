#include "MRCube.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRCube.h"
#include "MRMesh/MRMesh.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( Vector3f )

MRMesh* mrMakeCube( const MRVector3f* size_, const MRVector3f* base_ )
{
    ARG( size ); ARG( base );
    RETURN_NEW( makeCube( size, base ) );
}
