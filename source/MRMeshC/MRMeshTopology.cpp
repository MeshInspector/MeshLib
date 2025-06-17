#include "MRMeshTopology.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMeshTopology.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeBitSet )
REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( MeshTopology )
REGISTER_AUTO_CAST( ThreeVertIds )
REGISTER_AUTO_CAST( VertBitSet )
REGISTER_AUTO_CAST( VertId )
REGISTER_AUTO_CAST( FaceId )
REGISTER_VECTOR( Triangulation )
REGISTER_VECTOR( EdgePath )

void mrMeshTopologyPack( MRMeshTopology* top_ )
{
    ARG( top );
    top.pack();
}

const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top_ )
{
    ARG( top );
    RETURN( &top.getValidVerts() );
}

const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top_ )
{
    ARG( top );
    RETURN( &top.getValidFaces() );
}

MR_VECTOR_LIKE_IMPL( Triangulation, ThreeVertIds )

MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top_ )
{
    ARG( top );
    RETURN_NEW_VECTOR( top.getTriangulation() );
}

MR_VECTOR_LIKE_IMPL( EdgePath, EdgeId )

MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top_ )
{
    ARG( top );
    RETURN_NEW_VECTOR( top.findHoleRepresentiveEdges() );
}

int mrMeshTopologyFindNumHoles( const MRMeshTopology* top_, MREdgeBitSet* holeRepresentativeEdges_ )
{
    ARG( top ); ARG_PTR( holeRepresentativeEdges );
    return top.findNumHoles( holeRepresentativeEdges );
}

size_t mrMeshTopologyFaceSize( const MRMeshTopology* top_ )
{
    ARG( top );
    return top.faceSize();
}

void mrMeshTopologyGetLeftTriVerts( const MRMeshTopology* top_, MREdgeId a_, MRVertId* v0_, MRVertId* v1_, MRVertId* v2_ )
{
    ARG( top ); ARG_VAL( a ); ARG( v0 ); ARG( v1 ); ARG( v2 );
    top.getLeftTriVerts( a, v0, v1, v2 );
}

void mrMeshTopologyGetTriVerts( const MRMeshTopology* top_, MRFaceId f_, MRVertId* v0_, MRVertId* v1_, MRVertId* v2_ )
{
    ARG( top ); ARG_VAL( f ); ARG( v0 ); ARG( v1 ); ARG( v2 );
    top.getTriVerts( f, v0, v1, v2 );
}
