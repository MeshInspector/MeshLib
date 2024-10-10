#include "MRMeshCollidePrecise.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMeshCollidePrecise.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( ConvertToFloatVector )
REGISTER_AUTO_CAST( ConvertToIntVector )
REGISTER_AUTO_CAST( EdgeTri )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( PreciseCollisionResult )
REGISTER_VECTOR_LIKE( MRVectorEdgeTri, EdgeTri )

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

static_assert( sizeof( MREdgeTri ) == sizeof( EdgeTri ) );

bool mrEdgeTriEq( const MREdgeTri* a_, const MREdgeTri* b_ )
{
    ARG( a ); ARG( b );
    return a == b;
}

MR_VECTOR_IMPL( EdgeTri )

MRPreciseCollisionResult* mrFindCollidingEdgeTrisPrecise( const MRMeshPart* a, const MRMeshPart* b, const MRConvertToIntVector* conv_, const MRAffineXf3f* rigidB2A_, bool anyIntersection )
{
    ARG( conv ); ARG_PTR( rigidB2A );
    RETURN_NEW( findCollidingEdgeTrisPrecise(
        cast( *a ),
        cast( *b ),
        conv,
        rigidB2A,
        anyIntersection
    ) );
}

MRCoordinateConverters mrGetVectorConverters( const MRMeshPart* a, const MRMeshPart* b, const MRAffineXf3f* rigidB2A_ )
{
    ARG_PTR( rigidB2A );
    auto result = getVectorConverters( cast( *a ), cast( *b ), rigidB2A );
    return {
        .toInt = auto_cast( new_from( std::move( result.toInt ) ) ),
        .toFloat = auto_cast( new_from( std::move( result.toFloat ) ) ),
    };
}

const MRVectorEdgeTri mrPreciseCollisionResultEdgesAtrisB( const MRPreciseCollisionResult* result_ )
{
    ARG( result );
    RETURN_VECTOR( result.edgesAtrisB );
}

const MRVectorEdgeTri mrPreciseCollisionResultEdgesBtrisA( const MRPreciseCollisionResult* result_ )
{
    ARG( result );
    RETURN_VECTOR( result.edgesBtrisA );
}

void mrPreciseCollisionResultFree( MRPreciseCollisionResult* result_ )
{
    ARG_PTR( result );
    delete result;
}
