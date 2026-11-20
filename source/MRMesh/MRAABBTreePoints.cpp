#include "MRAABBTreePoints.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include "MRMesh.h"
#include "MRHeapBytes.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRPch/MRTBB.h"
#include <tbb/parallel_invoke.h>
#include <algorithm>
#include <unordered_map>

namespace MR
{

namespace
{

using Point = AABBTreePoints::Point;
using SimdNode = AABBTreePoints::SimdNode;
constexpr int cLeaf = AABBTreePoints::MaxNumPointsInLeaf;
/// the nodes of at least this many points are made in parallel
constexpr int cParallelPoints = 8192;

/// the number of points in the first half of a binary node of n > cLeaf points: the middle rounded up to whole leaves
inline int firstHalf( int n )
{
    const int m = n / 2;
    return m + ( cLeaf - m % cLeaf ) % cLeaf;
}

/// the shape of the tree, which depends only on the number of points: the tree is the binary tree of halves
/// (each split at firstHalf), with the binary levels grouped by two from the leaves up;
/// so a binary node of an odd height is not a node of this tree, and its children are in the slots of its parent
class Shape
{
public:
    /// fills the cache for all nodes of the tree of n points; afterwards it is only read, so the parallel build can share it
    explicit Shape( int n ) { height( n ); count( n ); }

    /// the sizes of the children of a node of n > cLeaf points
    int slots( int n, int ( &s )[4] ) const
    {
        int ns = 0;
        const int a = firstHalf( n );
        for ( int h : { a, n - a } )
        {
            if ( expand( h ) )
            {
                const int q = firstHalf( h );
                s[ns++] = q;
                s[ns++] = h - q;
            }
            else
                s[ns++] = h;
        }
        return ns;
    }

    /// whether the binary node of n points is skipped, and its children are put in the slots of its parent
    bool expand( int n ) const { return n > cLeaf && info_.at( n ).height % 2 != 0; }

    /// the number of nodes in the subtree of the node of n > cLeaf points, including itself
    int count( int n ) const { return info_.at( n ).count; }

private:
    struct Info { int height = 0; int count = 0; };
    std::unordered_map<int, Info> info_;

    int height( int n )
    {
        if ( n <= cLeaf )
            return 0;
        if ( auto it = info_.find( n ); it != info_.end() )
            return it->second.height;
        const int a = firstHalf( n );
        const int h = 1 + std::max( height( a ), height( n - a ) );
        info_[n].height = h;
        return h;
    }

    int count( int n )
    {
        int s[4];
        const int ns = slots( n, s );
        int c = 1;
        for ( int i = 0; i < ns; ++i )
            if ( s[i] > cLeaf )
                c += count( s[i] );
        info_[n].count = c;
        return c;
    }
};

/// splits each node in two (or four) at the median of the points along the largest dimension of their box,
/// as the binary tree did before, so the points of each leaf are the same
class Maker
{
public:
    Maker( std::vector<Point> & points, Vector<SimdNode> & nodes, const Shape & shape ) : points_( points ), nodes_( nodes ), shape_( shape ) {}

    Box3f boxOf( int first, int n ) const
    {
        Box3f box;
        for ( int i = first; i < first + n; ++i )
            box.include( points_[i].coord );
        return box;
    }

    /// restores the original order of the points of a leaf
    void sortLeaf( int first, int n )
    {
        std::sort( points_.data() + first, points_.data() + first + n, [] ( const Point & a, const Point & b ) { return a.id < b.id; } );
    }

    /// makes the node with given index of the points [first, first + n) having given box, and all its subtree
    void make( int idx, int first, int n, const Box3f & box )
    {
        struct Slot { int first = 0; int n = 0; Box3f box; };
        Slot halfSlots[2][2];
        const int a = split_( box, first, n );
        // the boxes of the halves, and the splits of the expanded ones
        auto makeHalf = [&] ( int k )
        {
            const int hf = k ? first + a : first;
            const int hn = k ? n - a : a;
            const Box3f hb = boxOf( hf, hn );
            if ( shape_.expand( hn ) )
            {
                const int q = split_( hb, hf, hn );
                halfSlots[k][0] = { hf, q, boxOf( hf, q ) };
                halfSlots[k][1] = { hf + q, hn - q, boxOf( hf + q, hn - q ) };
            }
            else
                halfSlots[k][0] = { hf, hn, hb };
        };
        if ( n >= cParallelPoints )
            tbb::parallel_invoke( [&] { makeHalf( 0 ); }, [&] { makeHalf( 1 ); } );
        else
        {
            makeHalf( 0 );
            makeHalf( 1 );
        }
        Slot slots[4];
        int ns = 0;
        for ( const auto & hs : halfSlots )
            for ( const auto & sl : hs )
                if ( sl.n > 0 )
                    slots[ns++] = sl;

        int childIdx[4];
        SimdNode & node = nodes_[idx];
        int next = idx + 1;
        for ( int i = 0; i < 4; ++i )
        {
            if ( i >= ns )
            {
                node.setEmpty( i );
                node.child[i] = 0;
                continue;
            }
            const auto & s = slots[i];
            node.set( i, s.box );
            if ( s.n > cLeaf )
            {
                node.child[i] = childIdx[i] = next;
                next += shape_.count( s.n );
            }
            else
            {
                node.child[i] = ~( s.first / cLeaf );
                childIdx[i] = -1;
            }
        }
        assert( next == idx + shape_.count( n ) );

        auto makeSlot = [&] ( int i )
        {
            const auto & s = slots[i];
            if ( childIdx[i] >= 0 )
                make( childIdx[i], s.first, s.n, s.box );
            else
                sortLeaf( s.first, s.n );
        };
        if ( n >= cParallelPoints )
            tbb::parallel_for( 0, ns, makeSlot );
        else
            for ( int i = 0; i < ns; ++i )
                makeSlot( i );
    }

private:
    std::vector<Point> & points_;
    Vector<SimdNode> & nodes_;
    const Shape & shape_;

    /// partitions the points [first, first + n) at firstHalf( n ) and returns it
    int split_( const Box3f & box, int first, int n )
    {
        assert( n > cLeaf && first % cLeaf == 0 );
        // define total order of Points: no two distinct points are equivalent
        auto less = [sortedDims = findSortedBoxDims( box )]( const Point & a, const Point & b )
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
        const int a = firstHalf( n );
        std::nth_element( points_.data() + first, points_.data() + first + a, points_.data() + first + n, less );
        return a;
    }
};

} // anonymous namespace

static void makeTree( const VertCoords & points, const VertBitSet * validPoints,
    std::vector<Point> & orderedPoints, Vector<SimdNode> & nodes, Box3f & box )
{
    MR_TIMER;

    const int numPoints = validPoints ? int( validPoints->count() ) : int( points.size() );
    if ( numPoints <= 0 )
        return;

    orderedPoints.resize( numPoints );
    int n = 0;
    if ( validPoints )
    {
        for ( auto v : *validPoints )
            orderedPoints[n++] = { points[v], v };
    }
    else
    {
        for ( auto v = 0_v; v < points.size(); ++v )
            orderedPoints[n++] = { points[v], v };
    }

    const Shape shape( numPoints );
    Maker maker( orderedPoints, nodes, shape );
    box = maker.boxOf( 0, numPoints );
    if ( numPoints <= cLeaf )
    {
        maker.sortLeaf( 0, numPoints );
        return;
    }
    nodes.resizeNoInit( size_t( shape.count( numPoints ) ) );
    maker.make( 0, 0, numPoints, box );
}

AABBTreePoints::AABBTreePoints( const PointCloud& pointCloud )
{
    makeTree( pointCloud.points, &pointCloud.validPoints, orderedPoints_, simdNodes_, box_ );
}

AABBTreePoints::AABBTreePoints( const Mesh& mesh )
{
    makeTree( mesh.points, &mesh.topology.getValidVerts(), orderedPoints_, simdNodes_, box_ );
}

AABBTreePoints::AABBTreePoints( const VertCoords & points, const VertBitSet * validPoints )
{
    makeTree( points, validPoints, orderedPoints_, simdNodes_, box_ );
}

void AABBTreePoints::getLeafOrder( VertBMap & vertMap ) const
{
    MR_TIMER;
    VertId newId = 0_v;
    for ( const auto & p : orderedPoints_ )
        vertMap.b[p.id] = newId++;
    vertMap.tsize = int( newId );
}

void AABBTreePoints::getLeafOrderAndReset( VertBMap& vertMap )
{
    MR_TIMER;
    VertId newId = 0_v;
    for ( auto & p : orderedPoints_ )
    {
        vertMap.b[p.id] = newId;
        p.id = newId++;
    }
    vertMap.tsize = int( newId );
}

size_t AABBTreePoints::heapBytes() const
{
    return simdNodes_.heapBytes() + MR::heapBytes( orderedPoints_ );
}

void AABBTreePoints::refit( const VertCoords & newCoords, const VertBitSet * changedVerts )
{
    MR_TIMER;
    if ( orderedPoints_.empty() )
        return;

    // updates the changed points of the leaf, and returns its new box if any point has changed
    auto refitLeaf = [&] ( int leaf, Box3f & box )
    {
        const auto [first, last] = leafPointRange( leaf );
        bool changed = false;
        for ( int i = first; i < last; ++i )
        {
            auto & p = orderedPoints_[i];
            if ( !changedVerts || changedVerts->test( p.id ) )
            {
                p.coord = newCoords[p.id];
                changed = true;
            }
            else
                assert( p.coord == newCoords[p.id] );
        }
        if ( !changed )
            return false;
        box = Box3f();
        for ( int i = first; i < last; ++i )
            box.include( orderedPoints_[i].coord );
        return true;
    };

    if ( simdNodes_.empty() )
    {
        Box3f box;
        if ( refitLeaf( 0, box ) )
            box_ = box;
        return;
    }

    // updates the subtree of the node, and returns true if its box has changed; the top levels are updated in parallel
    auto refitNode = [&] ( int n, int depth, auto && self ) -> bool
    {
        auto & node = simdNodes_[n];
        bool changed[MaxChildren] = {};
        auto refitChild = [&] ( int i )
        {
            const int c = node.child[i];
            if ( c < 0 )
            {
                Box3f box;
                if ( ( changed[i] = refitLeaf( ~c, box ) ) )
                    node.set( i, box );
            }
            else if ( ( changed[i] = self( c, depth + 1, self ) ) )
                node.set( i, simdNodes_[c].getUnion() );
        };
        const int numCh = 1 + ( node.child[1] != 0 ) + ( node.child[2] != 0 ) + ( node.child[3] != 0 );
        if ( depth < 4 )
            tbb::parallel_for( 0, numCh, refitChild );
        else
            for ( int i = 0; i < numCh; ++i )
                refitChild( i );
        return changed[0] || changed[1] || changed[2] || changed[3];
    };
    if ( refitNode( 0, 0, refitNode ) )
        box_ = simdNodes_[0].getUnion();
}

} // namespace MR
