#include "MRObjectMeshData.h"
#include "MRHeapBytes.h"
#include "MRColor.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"
#include "MRPch/MRSpdlog.h"

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

template <typename T, typename I>
static void resizeAttribute( Vector<T, I> & attr, I lastValid, size_t size, const char * name, const char * elements )
{
    if ( attr.empty() )
        return;
    if ( lastValid && attr.size() <= lastValid )
        spdlog::warn( "{} has {} elements for {} mesh {}, padding it with default values", name, attr.size(), size, elements );
    attr.resize( size );
}

void resizeAttributesToMesh( ObjectMeshData & data )
{
    if ( !data.mesh )
    {
        assert( false );
        return;
    }
    const auto & topology = data.mesh->topology;
    resizeAttribute( data.uvCoordinates, topology.lastValidVert(), topology.vertSize(), "uvCoordinates", "vertices" );
    resizeAttribute( data.vertColors, topology.lastValidVert(), topology.vertSize(), "vertColors", "vertices" );
    resizeAttribute( data.faceColors, topology.lastValidFace(), topology.faceSize(), "faceColors", "faces" );
    resizeAttribute( data.texturePerFace, topology.lastValidFace(), topology.faceSize(), "texturePerFace", "faces" );
}

} //namespace MR
