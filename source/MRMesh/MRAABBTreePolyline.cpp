#include "MRAABBTreePolyline.h"
#include "MRAABBTreeMaker.h"
#include "MRPolyline.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"

namespace MR
{

template<typename V>
AABBTreePolyline<V>::AABBTreePolyline( const typename PolylineTraits<V>::Polyline & polyline )
{
    MR_TIMER;

    using BoxedLine = BoxedLeaf<Traits>;
    std::vector<BoxedLine> boxedLines;
    boxedLines.resize( polyline.topology.undirectedEdgeSize() );

    int numLines = 0;
    for ( UndirectedEdgeId ue{ 0 }; ue < polyline.topology.undirectedEdgeSize(); ++ue )
    {
        if ( !polyline.topology.isLoneEdge( ue ) )
            boxedLines[numLines++].leafId = ue;
    }
    boxedLines.resize( numLines );
    if ( numLines <= 0 )
        return;

    // compute aabb's of each line
    tbb::parallel_for( tbb::blocked_range<int>( 0, numLines ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            const auto e = boxedLines[i].leafId;
            boxedLines[i].box.include( polyline.orgPnt( e ) );
            boxedLines[i].box.include( polyline.destPnt( e ) );
        }
    } );

    nodes_ = makeAABBTreeNodeVec( std::move( boxedLines ) );
}

template<typename V>
AABBTreePolyline<V>::AABBTreePolyline( const Mesh& mesh, const UndirectedEdgeBitSet & edgeSet )
{
    MR_TIMER;

    using BoxedLine = BoxedLeaf<Traits>;
    std::vector<BoxedLine> boxedLines;
    boxedLines.reserve( edgeSet.count() );

    for ( auto ue : edgeSet )
        boxedLines.push_back( { ue } );
    int numLines = (int)boxedLines.size();
    if ( numLines <= 0 )
        return;

    // compute aabb's of each line
    tbb::parallel_for( tbb::blocked_range<int>( 0, numLines ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            const auto e = boxedLines[i].leafId;
            boxedLines[i].box.include( mesh.orgPnt( e ) );
            boxedLines[i].box.include( mesh.destPnt( e ) );
        }
    } );

    nodes_ = makeAABBTreeNodeVec( std::move( boxedLines ) );
}

template AABBTreePolyline<Vector2f>::AABBTreePolyline( const Polyline2 & );
template AABBTreePolyline<Vector3f>::AABBTreePolyline( const Polyline3 & );
template AABBTreePolyline<Vector3f>::AABBTreePolyline( const Mesh &, const UndirectedEdgeBitSet & );

} //namespace MR
