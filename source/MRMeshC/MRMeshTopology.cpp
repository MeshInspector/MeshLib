#include "MRMeshTopology.h"

#include "MRMesh/MRMeshTopology.h"

using namespace MR;

void mrMeshTopologyPack( MRMeshTopology* top )
{
    reinterpret_cast<MeshTopology*>( top )->pack();
}

const MRBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top )
{
    return reinterpret_cast<const MRBitSet*>( &reinterpret_cast<const MeshTopology*>( top )->getValidVerts() );
}

const MRBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top )
{
    return reinterpret_cast<const MRBitSet*>( &reinterpret_cast<const MeshTopology*>( top )->getValidFaces() );
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
