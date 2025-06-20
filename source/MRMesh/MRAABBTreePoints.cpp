#include "MRAABBTreePoints.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRHeapBytes.h"
#include "MRBuffer.h"
#include <stack>
#include <thread>

namespace MR
{

struct SubtreePoints
{
    SubtreePoints( NodeId root, int f, int n ) : root( root ), firstPoint( f ), numPoints( n )
    {
    }
    NodeId root; // of subtree
    int firstPoint = 0;
    int numPoints = 0;
    NodeId lastNode() const
    {
        return root + getNumNodesPoints( numPoints );
    }
    bool leaf() const
    {
        assert( numPoints >= 1 );
        return numPoints <= AABBTreePoints::MaxNumPointsInLeaf;
    }
};

class AABBTreePointsMaker
{
public:
    std::pair<AABBTreePoints::NodeVec,std::vector<AABBTreePoints::Point>> construct(
        const VertCoords & points, const VertBitSet * validPoints );

private:
    std::vector<AABBTreePoints::Point> orderedPoints_;
    AABBTreePoints::NodeVec nodes_;

private:
    // [firstPoint, result) will go to left child and [result, lastPoint) - to the right child
    int partitionPoints( const Box3f& box, int firstPoint, int lastPoint );
    // constructs not-leaf node
    std::pair<SubtreePoints, SubtreePoints> makeNode( const SubtreePoints& s );
    // constructs given subtree, optionally splitting the job on given number of threads
    void makeSubtree( const SubtreePoints& s, int numThreads );
};

int AABBTreePointsMaker::partitionPoints( const Box3f& box, int firstPoint, int lastPoint )
{
    assert( firstPoint + AABBTreePoints::MaxNumPointsInLeaf < lastPoint );

    // define total order of Points: no two distinct points are equivalent
    auto less = [sortedDims = findSortedBoxDims( box )]( const AABBTreePoints::Point & a, const AABBTreePoints::Point & b )
    {
        // first compare (and split later) by the largest box dimension
        for ( int i = decltype( sortedDims )::elements - 1; i >= 0; --i )
        {
            const int splitDim = sortedDims[i];
            const auto aDim = a.coord[splitDim];
            const auto bDim = b.coord[splitDim];
            if ( aDim != bDim )
                return aDim < bDim;
        }
        // if two points have equal coordinates then compare by id to distinguish them
        return a.id < b.id;
    };

    int midPoint = firstPoint + ( lastPoint - firstPoint ) / 2;
    // to minimize the total number of nodes
    midPoint += ( AABBTreePoints::MaxNumPointsInLeaf - ( midPoint % AABBTreePoints::MaxNumPointsInLeaf ) ) % AABBTreePoints::MaxNumPointsInLeaf;
    std::nth_element( orderedPoints_.data() + firstPoint, orderedPoints_.data() + midPoint, orderedPoints_.data() + lastPoint, less );
    return midPoint;
}

std::pair<SubtreePoints, SubtreePoints> AABBTreePointsMaker::makeNode( const SubtreePoints& s )
{
    assert( !s.leaf() );
    auto& node = nodes_[s.root];
    assert( !node.box.valid() );
    for ( size_t i = 0; i < s.numPoints; ++i )
        node.box.include( orderedPoints_[s.firstPoint + i].coord );

    const int midPoint = partitionPoints( node.box, s.firstPoint, s.firstPoint + s.numPoints );
    const int leftNumPoints = midPoint - s.firstPoint;
    const int rightNumPoints = s.numPoints - leftNumPoints;
    node.l = s.root + 1;
    node.r = s.root + 1 + getNumNodesPoints( leftNumPoints );
    return
    {
        SubtreePoints( node.l, s.firstPoint, leftNumPoints ),
        SubtreePoints( node.r, midPoint,     rightNumPoints )
    };
}

void AABBTreePointsMaker::makeSubtree( const SubtreePoints& s, int numThreads )
{
    assert( s.root + getNumNodesPoints( s.numPoints ) <= nodes_.size() );

    if ( numThreads >= 2 && s.numPoints > 3 * AABBTreePoints::MaxNumPointsInLeaf )
    {
        const auto& lr = makeNode( s );
        const int rThreads = numThreads / 2;
        const int lThreads = numThreads - rThreads;
        tbb::task_group group;
        group.run( [&] () { makeSubtree( lr.second, rThreads ); } );
        makeSubtree( lr.first, lThreads );
        group.wait();
        return;
    }

    std::stack<SubtreePoints> stack;
    stack.push( s );

    while ( !stack.empty() )
    {
        const SubtreePoints x = stack.top();
        stack.pop();
        if ( x.leaf() )
        {
            // restore original vertex ordering within each leaf
            std::sort( orderedPoints_.data() + x.firstPoint, orderedPoints_.data() + x.firstPoint + x.numPoints,
                [&]( const AABBTreePoints::Point& a, const AABBTreePoints::Point& b )
            {
                return a.id < b.id;
            } );
            auto& node = nodes_[x.root];
            node.setLeafPointRange( x.firstPoint, x.firstPoint + x.numPoints );
            assert( !node.box.valid() );
            for ( size_t i = 0; i < x.numPoints; ++i )
                node.box.include( orderedPoints_[x.firstPoint + i].coord );
            continue;
        }

        const auto& [ls, rs] = makeNode( x );
        assert( ls.root < rs.root );
        stack.push( rs );
        stack.push( ls ); // to process it first
    }
}

std::pair<AABBTreePoints::NodeVec, std::vector<AABBTreePoints::Point>> AABBTreePointsMaker::construct(
    const VertCoords & points, const VertBitSet * validPoints )
{
    MR_TIMER;

    const int numPoints = validPoints ? int( validPoints->count() ) : int( points.size() );
    if ( numPoints <= 0 )
        return {};

    orderedPoints_.resize( numPoints );
    int n = 0;
    if ( validPoints )
    {
        for ( auto v : *validPoints )
            orderedPoints_[n++] = { points[v], v };
    }
    else
    {
        for ( auto v = 0_v; v < points.size(); ++v )
            orderedPoints_[n++] = { points[v], v };
    }

    nodes_.resize( getNumNodesPoints( numPoints ) );
    makeSubtree( SubtreePoints( AABBTreePoints::rootNodeId(), 0, numPoints ), std::thread::hardware_concurrency() );

    return {std::move( nodes_ ),std::move( orderedPoints_ )};
}

AABBTreePoints::AABBTreePoints( const PointCloud& pointCloud )
{
    auto [nodes, orderedPoints] = AABBTreePointsMaker().construct( pointCloud.points, &pointCloud.validPoints );
    nodes_ = std::move( nodes );
    orderedPoints_ = std::move( orderedPoints );
}

AABBTreePoints::AABBTreePoints( const Mesh& mesh )
{
    auto [nodes, orderedPoints] = AABBTreePointsMaker().construct( mesh.points, &mesh.topology.getValidVerts() );
    nodes_ = std::move( nodes );
    orderedPoints_ = std::move( orderedPoints );
}

AABBTreePoints::AABBTreePoints( const VertCoords & points, const VertBitSet * validPoints )
{
    auto [nodes, orderedPoints] = AABBTreePointsMaker().construct( points, validPoints );
    nodes_ = std::move( nodes );
    orderedPoints_ = std::move( orderedPoints );
}

void AABBTreePoints::getLeafOrder( VertBMap & vertMap ) const
{
    MR_TIMER;
    VertId newId = 0_v;
    for ( auto & n : nodes_ )
    {
        if ( !n.leaf() )
            continue;
        const auto [first, last] = n.getLeafPointRange();
        for ( int i = first; i < last; ++i )
        {
            auto oldId = orderedPoints_[i].id;
            vertMap.b[oldId] = newId++;
        }
    }
    vertMap.tsize = int( newId );
}

void AABBTreePoints::getLeafOrderAndReset( VertBMap& vertMap )
{
    MR_TIMER;
        VertId newId = 0_v;
    for ( auto& n : nodes_ )
    {
        if ( !n.leaf() )
            continue;
        const auto [first, last] = n.getLeafPointRange();
        for ( int i = first; i < last; ++i )
        {
            auto & id = orderedPoints_[i].id;
            vertMap.b[id] = newId;
            id = newId++;
        }
    }
    vertMap.tsize = int( newId );
}

size_t AABBTreePoints::heapBytes() const
{
    return
        nodes_.heapBytes() +
        MR::heapBytes( orderedPoints_ );
}

void AABBTreePoints::refit( const VertCoords & newCoords, const VertBitSet & changedVerts )
{
    MR_TIMER;

    // find changed orderedPoints_ and update them
    BitSet changedPoints( orderedPoints_.size() );
    BitSetParallelForAll( changedPoints, [&]( size_t i )
    {
        auto & p = orderedPoints_[i];
        if ( changedVerts.test( p.id ) )
        {
            changedPoints.set( i );
            p.coord = newCoords[p.id];
        }
        else
            assert( p.coord == newCoords[p.id] );
    } );

    // update leaf nodes
    NodeBitSet changedNodes( nodes_.size() );
    BitSetParallelForAll( changedNodes, [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        if ( !node.leaf() )
            return;
        bool changed = false;
        const auto [first, last] = node.getLeafPointRange();
        for ( int i = first; i < last; ++i )
            if ( changedPoints.test(i) )
            {
                changed = true;
                break;
            }
        if ( !changed )
            return;
        changedNodes.set( nid );
        Box3f box;
        for ( int i = first; i < last; ++i )
            box.include( orderedPoints_[i].coord );
        node.box = box;
    } );

    //update not-leaf nodes
    for ( auto nid = nodes_.backId(); nid; --nid )
    {
        auto & node = nodes_[nid];
        if ( node.leaf() )
            continue;
        if ( !changedNodes.test( node.l ) && !changedNodes.test( node.r ) )
            continue;
        changedNodes.set( nid );
        node.box = nodes_[node.l].box;
        node.box.include( nodes_[node.r].box );
    }
}

}
