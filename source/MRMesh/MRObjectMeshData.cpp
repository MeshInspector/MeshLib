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

template <typename T>
static UndirectedEdgeBitSet edgesBetweenDifferentValues( const MeshTopology & topology, const Vector<T, FaceId> & values )
{
    UndirectedEdgeBitSet res;
    if ( values.empty() )
        return res;
    res.resize( topology.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        EdgeId e( ue );
        auto l = topology.left( e );
        auto r = topology.right( e );
        if ( l < values.size() && r < values.size() && values[l] != values[r] )
            res.set( ue );
    } );
    return res;
}

UndirectedEdgeBitSet edgesBetweenDifferentColors( const MeshTopology & topology, const FaceColors & colors )
{
    MR_TIMER;
    return edgesBetweenDifferentValues( topology, colors );
}

UndirectedEdgeBitSet edgesBetweenDifferentTextures( const MeshTopology & topology, const TexturePerFace & textures )
{
    MR_TIMER;
    return edgesBetweenDifferentValues( topology, textures );
}

void resizeAttributesToMesh( ObjectMeshData & data )
{
    if ( !data.mesh )
    {
        assert( false );
        return;
    }
    auto resize = [] ( auto & attr, size_t size )
    {
        if ( !attr.empty() )
            attr.resize( size );
    };
    const auto & topology = data.mesh->topology;
    resize( data.uvCoordinates, topology.vertSize() );
    resize( data.vertColors, topology.vertSize() );
    resize( data.faceColors, topology.faceSize() );
    // not TextureId(), which is invalid: faces without texture id are rendered with the first texture
    if ( !data.texturePerFace.empty() )
        data.texturePerFace.resize( topology.faceSize(), TextureId{ 0 } );
}

} //namespace MR
