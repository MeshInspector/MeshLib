#include "MRMeshTopology.h"

#include "MRMesh/MRMeshTopology.h"

using namespace MR;

void mrMeshTopologyPack( MRMeshTopology* top )
{
    reinterpret_cast<MeshTopology*>( top )->pack();
}

const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top )
{
    return reinterpret_cast<const MRVertBitSet*>( &reinterpret_cast<const MeshTopology*>( top )->getValidVerts() );
}

const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top )
{
    return reinterpret_cast<const MRFaceBitSet*>( &reinterpret_cast<const MeshTopology*>( top )->getValidFaces() );
}

MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top )
{
    auto* tris = new Triangulation;
    *tris = reinterpret_cast<const MeshTopology*>( top )->getTriangulation();
    return reinterpret_cast<MRTriangulation*>( tris );
}

const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris )
{
    return reinterpret_cast<const MRThreeVertIds*>( reinterpret_cast<const Triangulation*>( tris )->data() );
}

size_t mrTriangulationSize( const MRTriangulation* tris )
{
    return reinterpret_cast<const Triangulation*>( tris )->size();
}

void mrTriangulationFree( MRTriangulation* tris )
{
    delete reinterpret_cast<Triangulation*>( tris );
}

MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top_ )
{
    const auto& top = *reinterpret_cast<const MeshTopology*>( top_ );

    auto* res = new std::vector<EdgeId>( top.findHoleRepresentiveEdges() );
    return reinterpret_cast<MREdgePath*>( res );
}

const MREdgeId* mrEdgePathData( const MREdgePath* ep_ )
{
    const auto& ep = *reinterpret_cast<const std::vector<EdgeId>*>( ep_ );
    return reinterpret_cast<const MREdgeId*>( ep.data() );
}

size_t mrEdgePathSize( const MREdgePath* ep_ )
{
    const auto& ep = *reinterpret_cast<const std::vector<EdgeId>*>( ep_ );
    return ep.size();
}

void mrEdgePathFree( MREdgePath* ep )
{
    delete reinterpret_cast<std::vector<EdgeId>*>( ep );
}
