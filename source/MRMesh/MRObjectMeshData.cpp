#include "MRObjectMeshData.h"
#include "MRHeapBytes.h"
#include "MRColor.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"

namespace MR
{

ObjectMeshData ObjectMeshData::clone() const
{
    ObjectMeshData res = *this;
    if ( res.mesh )
        res.mesh = std::make_shared<Mesh>( *res.mesh );
    return res;
}

size_t ObjectMeshData::heapBytes() const
{
    return MR::heapBytes( mesh )
        + selectedFaces.heapBytes()
        + selectedEdges.heapBytes()
        + creases.heapBytes()
        + vertColors.heapBytes()
        + faceColors.heapBytes()
        + uvCoordinates.heapBytes()
        + texturePerFace.heapBytes();
}

UndirectedEdgeBitSet edgesBetweenDifferentColors( const MeshTopology & topology, const FaceColors & colors )
{
    MR_TIMER;
    UndirectedEdgeBitSet res;
    if ( colors.empty() )
        return res;
    res.resize( topology.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        EdgeId e( ue );
        auto l = topology.left( e );
        auto r = topology.right( e );
        if ( l < colors.size() && r < colors.size() && colors[l] != colors[r] )
            res.set( ue );
    } );
    return res;
}

} //namespace MR
