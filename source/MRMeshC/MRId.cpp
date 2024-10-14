#include "MRId.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRId.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( UndirectedEdgeId )

static_assert( sizeof( MREdgeId ) == sizeof( EdgeId ) );
static_assert( sizeof( MRUndirectedEdgeId ) == sizeof( UndirectedEdgeId ) );
static_assert( sizeof( MRFaceId ) == sizeof( FaceId ) );
static_assert( sizeof( MRVertId ) == sizeof( VertId ) );
static_assert( sizeof( MRObjId ) == sizeof( ObjId ) );

MREdgeId mrEdgeIdFromUndirectedEdgeId( MRUndirectedEdgeId u_ )
{
    ARG_VAL( u );
    RETURN( EdgeId( u ) );
}

MREdgeId mrEdgeIdSym( MREdgeId e_ )
{
    ARG_VAL( e );
    RETURN( e.sym() );
}

MRUndirectedEdgeId mrEdgeIdUndirected( MREdgeId e_ )
{
    ARG_VAL( e );
    RETURN( e.undirected() );
}
