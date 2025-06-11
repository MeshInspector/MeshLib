#include "MRMeshCollidePrecise.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMeshCollidePrecise.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( ConvertToFloatVector )
REGISTER_AUTO_CAST( ConvertToIntVector )
REGISTER_AUTO_CAST( EdgeTri )
REGISTER_AUTO_CAST( VarEdgeTri )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )
REGISTER_VECTOR_LIKE( MRVectorEdgeTri, EdgeTri )
REGISTER_VECTOR_LIKE( MRVectorVarEdgeTri, VarEdgeTri )

static_assert( sizeof( MRVarEdgeTri ) == sizeof( VarEdgeTri ) );

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
MR_VECTOR_IMPL( VarEdgeTri )

MRPreciseCollisionResult* mrFindCollidingEdgeTrisPrecise( const MRMeshPart* a, const MRMeshPart* b, const MRConvertToIntVector* conv_, const MRAffineXf3f* rigidB2A_, bool anyIntersection )
{
    ARG( conv ); ARG_PTR( rigidB2A );
    RETURN_NEW_VECTOR( findCollidingEdgeTrisPrecise(
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
