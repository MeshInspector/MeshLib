#include "MRMeshTopology.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshTopology.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( EdgePath )
REGISTER_AUTO_CAST( MeshTopology )
REGISTER_AUTO_CAST( ThreeVertIds )
REGISTER_AUTO_CAST( Triangulation )
REGISTER_AUTO_CAST( VertId )

void mrMeshTopologyPack( MRMeshTopology* top_ )
{
    ARG( top );
    top.pack();
}

const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top_ )
{
    ARG( top );
    return cast_to<MRVertBitSet>( &top.getValidVerts() );
}

const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top_ )
{
    ARG( top );
    return cast_to<MRFaceBitSet>( &top.getValidFaces() );
}

MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top_ )
{
    ARG( top );
    RETURN_NEW( top.getTriangulation() );
}

const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris_ )
{
    ARG( tris );
    RETURN( tris.data() );
}

size_t mrTriangulationSize( const MRTriangulation* tris_ )
{
    ARG( tris );
    return tris.size();
}

void mrTriangulationFree( MRTriangulation* tris_ )
{
    ARG_PTR( tris );
    delete tris;
}

MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top_ )
{
    ARG( top );
    RETURN_NEW( top.findHoleRepresentiveEdges() );
}

const MREdgeId* mrEdgePathData( const MREdgePath* ep_ )
{
    ARG( ep );
    RETURN( ep.data() );
}

size_t mrEdgePathSize( const MREdgePath* ep_ )
{
    ARG( ep );
    return ep.size();
}

void mrEdgePathFree( MREdgePath* ep_ )
{
    ARG_PTR( ep );
    delete ep;
}

int mrMeshTopologyFindNumHoles( const MRMeshTopology* top_, MREdgeBitSet* holeRepresentativeEdges_ )
{
    ARG( top ); ARG_PTR_OF( EdgeBitSet, holeRepresentativeEdges );
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
