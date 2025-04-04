#include "MRLaplacian.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRLaplacian.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeWeights )
REGISTER_AUTO_CAST( VertexMass )
REGISTER_AUTO_CAST( Laplacian )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( VertBitSet )
REGISTER_AUTO_CAST( VertId )
REGISTER_AUTO_CAST2( Laplacian::RememberShape, MRLaplacianRememberShape )

MRLaplacian* mrLaplacianNew( MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( Laplacian( mesh ) );
}

void mrLaplacianFree( MRLaplacian* laplacian_ )
{
    ARG_PTR( laplacian );
    delete laplacian;
}

void mrLaplacianInit( MRLaplacian* laplacian_, const MRVertBitSet* freeVerts_, MREdgeWeights weights_, MRVertexMass vmass_, MRLaplacianRememberShape rem_ )
{
    ARG( laplacian ); ARG( freeVerts ); ARG_VAL( weights ); ARG_VAL( vmass ); ARG_VAL( rem );
    laplacian.init( freeVerts, weights, vmass, rem );
}

void mrLaplacianFixVertex( MRLaplacian* laplacian_, MRVertId v_, const MRVector3f* fixedPos_, bool smooth )
{
    ARG( laplacian ); ARG_VAL( v ); ARG( fixedPos );
    laplacian.fixVertex( v, fixedPos, smooth );
}

void mrLaplacianApply( MRLaplacian* laplacian_ )
{
    ARG( laplacian );
    laplacian.apply();
}
