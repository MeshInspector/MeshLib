#include "MRSegmentMesh.h"
#include "MRMeshSegmenter.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRUnionFind.h"
#include "MRColor.h"
#include "MRHexPalette.h"
#include "MRTimer.h"

namespace MR
{

namespace
{

struct SegmentData
{
    double area = 0;
    SegmentData& operator +=( const SegmentData& r ) { area += r.area; return * this; }
};

struct SegmentBdData
{
    double length = 0;
    double lengthCurv = 0;
    SegmentBdData& operator +=( const SegmentBdData& r ) { length += r.length; lengthCurv += r.lengthCurv; return * this; }
};

} //anonymous namespace

Expected<GroupOrder> segmentMesh( const Mesh& mesh, const EdgeMetric& curvMetric, const ProgressCallback& progress )
{
    MR_TIMER;
    if ( !curvMetric )
        return unexpected( "no curvMetric given" );

    auto mergePenalty = []( const SegmentData& a, const SegmentData& b, const SegmentBdData& bd )
    {
        if ( bd.length <= 0 )
            return prohibitMergePenalty;
        return std::min( a.area, b.area ) * bd.lengthCurv / sqr( bd.length );
    };

    return MeshSegmenter<SegmentData, SegmentBdData, decltype( mergePenalty )>::run( mesh.topology,
        [&]( FaceId f )
        {
            return SegmentData{ .area = mesh.area( f ) };
        },
        [&]( UndirectedEdgeId ue )
        {
            const auto len = mesh.edgeLength( ue );
            return SegmentBdData{ .length = len, .lengthCurv = len * curvMetric( ue ) };
        },
        mergePenalty, progress );
}

UndirectedEdgeBitSet findSegmentBoundaries( const MeshTopology& topology,
    const GroupOrder& groupOrder, int numSegments, FaceColors* outFaceColors )
{
    MR_TIMER;
    int numMerges = topology.numValidFaces() - numSegments;
    assert( numMerges >= 0 );
    assert( numMerges <= groupOrder.size() );
    numMerges = std::clamp( numMerges, 0, (int)groupOrder.size() );

    UnionFind<FaceId> unionFindStruct( topology.faceSize() );
    for ( int i = 0; i < numMerges; ++i )
    {
       [[maybe_unused]] auto p = unionFindStruct.unite( groupOrder[i].aFace, groupOrder[i].bFace );
       assert( p.second );
    }
    const auto & roots = unionFindStruct.roots();

    UndirectedEdgeBitSet res( topology.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        const auto l = topology.left( ue );
        if ( !l )
            return;
        const auto r = topology.right( ue );
        if ( !r )
            return;
        if ( roots[l] != roots[r] )
            res.set( ue );
    } );

    if ( outFaceColors )
    {
        outFaceColors->resizeNoInit( topology.faceSize() );
        HashMap<FaceId, Color> root2Color;
        // give colors to segments ignoring the contrast on their boundaries
        int nextColor = 0;
        for ( auto f : topology.getValidFaces() )
        {
            if ( roots[f] != f )
                continue;
            root2Color[f] = HexPalette::colorAtStep( nextColor++ );
        }
        BitSetParallelFor( topology.getValidFaces(), [&]( FaceId& f )
        {
            auto it = root2Color.find( roots[f] );
            assert( it != root2Color.end() );
            (*outFaceColors)[f] = it->second;
        } );
    }

    return res;
}

} //namespace MR
