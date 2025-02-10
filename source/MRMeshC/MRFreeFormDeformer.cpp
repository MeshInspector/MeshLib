#include "MRFreeFormDeformer.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRFreeFormDeformer.h"

using namespace MR;

REGISTER_AUTO_CAST( Box3f )
REGISTER_AUTO_CAST( FreeFormDeformer )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( Vector3i )
REGISTER_AUTO_CAST( VertBitSet )

MRFreeFormDeformer* mrFreeFormDeformerNewFromMesh( MRMesh* mesh_, const MRVertBitSet* region_ )
{
    ARG( mesh ); ARG_PTR( region );
    RETURN_NEW( FreeFormDeformer( mesh, region ) );
}

void mrFreeFormDeformerFree( MRFreeFormDeformer* deformer_ )
{
    ARG_PTR( deformer );
    delete deformer;
}

void mrFreeFormDeformerInit( MRFreeFormDeformer* deformer_, const MRVector3i* resolution_, const MRBox3f* initialBox_ )
{
    ARG( deformer ); ARG( resolution ); ARG( initialBox );
    deformer.init( resolution, initialBox );
}

void mrFreeFormDeformerSetRefGridPointPosition( MRFreeFormDeformer* deformer_, const MRVector3i* coordOfPointInGrid_, const MRVector3f* newPos_ )
{
    ARG( deformer ); ARG( coordOfPointInGrid ); ARG( newPos );
    deformer.setRefGridPointPosition( coordOfPointInGrid, newPos );
}

MRVector3f mrFreeFormDeformerGetRefGridPointPosition( const MRFreeFormDeformer* deformer_, const MRVector3i* coordOfPointInGrid_ )
{
    ARG( deformer ); ARG( coordOfPointInGrid );
    RETURN( deformer.getRefGridPointPosition( coordOfPointInGrid ) );
}

void mrFreeFormDeformerApply( const MRFreeFormDeformer* deformer_ )
{
    ARG( deformer );
    deformer.apply();
}
