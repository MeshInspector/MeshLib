#include "MRAABBTreePolyline.h"
#include "MRAABBTreeBase.hpp"
#include "MRAABBTreeMaker.hpp"
#include "MRPolyline.h"
#include "MRMesh.h"
#include "MRBuffer.h"
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
    Buffer<BoxedLine> boxedLines;
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
            Box<V> box;
            box.include( polyline.orgPnt( e ) );
            box.include( polyline.destPnt( e ) );
            boxedLines[i].box = box;
        }
    } );

    nodes_ = makeAABBTreeNodeVec( std::move( boxedLines ) );
}

template<typename V>
AABBTreePolyline<V>::AABBTreePolyline( const Mesh& mesh, const UndirectedEdgeBitSet & edgeSet ) MR_REQUIRES_IF_SUPPORTED( V::elements == 3 )
{
    MR_TIMER;

    using BoxedLine = BoxedLeaf<Traits>;
    Buffer<BoxedLine> boxedLines;
    boxedLines.resize( edgeSet.count() );
    if ( boxedLines.size() <= 0 )
        return;

    int numLines = 0;
    for ( auto ue : edgeSet )
        boxedLines[numLines++].leafId = ue;

    // compute aabb's of each line
    tbb::parallel_for( tbb::blocked_range<int>( 0, numLines ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            const auto e = boxedLines[i].leafId;
            Box<V> box;
            box.include( mesh.orgPnt( e ) );
            box.include( mesh.destPnt( e ) );
            boxedLines[i].box = box;
        }
    } );

    nodes_ = makeAABBTreeNodeVec( std::move( boxedLines ) );
}

template AABBTreePolyline<Vector2f>::AABBTreePolyline( const Polyline2 & );
template AABBTreePolyline<Vector3f>::AABBTreePolyline( const Polyline3 & );
template AABBTreePolyline<Vector3f>::AABBTreePolyline( const Mesh &, const UndirectedEdgeBitSet & )
#ifdef _MSC_VER
MR_REQUIRES_IF_SUPPORTED( Vector3f::elements == 3 )
#endif
;

template auto AABBTreeBase<LineTreeTraits<Vector2f>>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<LineTreeTraits<Vector3f>>::getSubtrees( int minNum ) const -> std::vector<NodeId>;

template auto AABBTreeBase<LineTreeTraits<Vector2f>>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template auto AABBTreeBase<LineTreeTraits<Vector3f>>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;

template NodeBitSet AABBTreeBase<LineTreeTraits<Vector2f>>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template NodeBitSet AABBTreeBase<LineTreeTraits<Vector3f>>::getNodesFromLeaves( const LeafBitSet & leaves ) const;

template void AABBTreeBase<LineTreeTraits<Vector2f>>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<LineTreeTraits<Vector3f>>::getLeafOrder( LeafBMap & leafMap ) const;

template void AABBTreeBase<LineTreeTraits<Vector2f>>::getLeafOrderAndReset( LeafBMap & leafMap );
template void AABBTreeBase<LineTreeTraits<Vector3f>>::getLeafOrderAndReset( LeafBMap & leafMap );

} //namespace MR
