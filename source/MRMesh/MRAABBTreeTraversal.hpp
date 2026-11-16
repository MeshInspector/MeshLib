#pragma once

#include "MRAABBTree.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include "MRBuffer.h"
#include "MREnums.h"
#include "MRInplaceStack.h"
#include "MRSimd4.hpp"
#include "MRVector.h"
#include <atomic>
#include <cstdint>
#include <optional>

namespace MR
{

/// a pair of nodes of two AABBTree (or of one tree) in their simultaneous traversal
struct WideNodePair
{
    NodeId a, b;
};

/// returns the bit mask of the children of not-leaf node parent (bit i for child parent.first + i), whose boxes intersect given box
/// (as Box3f::intersects) by one SIMD test of all children
inline int childBoxesIntersect( const AABBTree & tree, NodeId parent, const Box3f & box )
{
    using namespace Simd4;
    const auto & sn = tree.simdNodes()[tree.simdRef( parent )];
    M ok = andm( le( load( sn.box[0] ), set1( box.max.x ) ), le( set1( box.min.x ), load( sn.box[3] ) ) );
    ok = andm( ok, andm( le( load( sn.box[1] ), set1( box.max.y ) ), le( set1( box.min.y ), load( sn.box[4] ) ) ) );
    ok = andm( ok, andm( le( load( sn.box[2] ), set1( box.max.z ) ), le( set1( box.min.z ), load( sn.box[5] ) ) ) );
    return bits( ok ); // unused slots have empty boxes
}

/// intersect( aNode, bNode ) of splitWidePair together with
/// children( aSide, parent, other ) returning the bit mask of the children of parent (in tree a if aSide, otherwise in tree b),
/// for which intersect with node other of the other tree would return true
template<typename Pair, typename Children>
struct WideIntersect
{
    Pair pair;
    Children children;
    bool operator()( NodeId a, NodeId b ) const { return pair( a, b ); }
};
template<typename Pair, typename Children>
WideIntersect( Pair, Children ) -> WideIntersect<Pair, Children>;

/// pushes the pairs of each child of the bigger non-leaf node with the other node, if intersect( aNode, bNode ) of their boxes;
/// returns false and pushes nothing if both nodes are leaves
template<typename Intersect, typename Push>
bool splitWidePair( const AABBTree & aTree, const AABBTree & bTree, const WideNodePair & s, Intersect && intersect, Push && push )
{
    const auto & an = aTree.nodes()[s.a];
    const auto & bn = bTree.nodes()[s.b];
    constexpr bool hasChildren = requires { intersect.children( true, s.a, s.b ); };
    if ( !an.leaf() && ( bn.leaf() || aTree.box( s.a ).volume() >= bTree.box( s.b ).volume() ) )
    {
        if constexpr ( hasChildren )
        {
            for ( int m = intersect.children( true, s.a, s.b ); m; m &= m - 1 )
                push( WideNodePair{ NodeId( an.first + Simd4::lowestBit( m ) ), s.b } );
        }
        else
        {
            for ( int i = 0; i < an.count; ++i )
                if ( const NodeId c( an.first + i ); intersect( c, s.b ) )
                    push( WideNodePair{ c, s.b } );
        }
        return true;
    }
    if ( !bn.leaf() )
    {
        if constexpr ( hasChildren )
        {
            for ( int m = intersect.children( false, s.b, s.a ); m; m &= m - 1 )
                push( WideNodePair{ s.a, NodeId( bn.first + Simd4::lowestBit( m ) ) } );
        }
        else
        {
            for ( int i = 0; i < bn.count; ++i )
                if ( const NodeId c( bn.first + i ); intersect( s.a, c ) )
                    push( WideNodePair{ s.a, c } );
        }
        return true;
    }
    return false;
}

/// the same for the traversal of one tree with itself: a node paired with itself gives each child paired with itself and the pairs of different children
template<typename Intersect, typename Push>
bool splitSelfWidePair( const AABBTree & tree, const WideNodePair & s, Intersect && intersect, Push && push )
{
    if ( s.a != s.b )
        return splitWidePair( tree, tree, s, intersect, push );
    const auto & n = tree.nodes()[s.a];
    if ( n.leaf() )
        return false;
    for ( int i = 0; i < n.count; ++i )
    {
        const NodeId ci( n.first + i );
        push( WideNodePair{ ci, ci } );
        if constexpr ( requires { intersect.children( true, s.a, s.b ); } )
        {
            // the children after ci, whose boxes intersect the box of ci
            for ( int m = intersect.children( true, s.a, ci ) & ( ~0 << ( i + 1 ) ); m; m &= m - 1 )
                push( WideNodePair{ ci, NodeId( n.first + Simd4::lowestBit( m ) ) } );
        }
        else
        {
            for ( int j = i + 1; j < n.count; ++j )
                if ( const NodeId cj( n.first + j ); intersect( ci, cj ) )
                    push( WideNodePair{ ci, cj } );
        }
    }
    return true;
}

/// the boxes of the triangles of given leaf in SoA order ( lo[c][i], hi[c][i] ), computed at once from the SIMD layout:
/// each contains tree.leafBox( i ) of its triangle (expanded by about two units in the last place more), so they can replace
/// these boxes in the tests, which only select the candidates for precise tests; empty (NaN) in unused slots
struct WideLeafTriBoxes
{
    alignas( 16 ) float lo[3][4];
    alignas( 16 ) float hi[3][4];

    WideLeafTriBoxes( const AABBTree & tree, NodeId leafNode )
    {
        using namespace Simd4;
        const auto & leaf = tree.simdLeaves()[~tree.simdRef( leafNode )];
        const F k = set1( 2.4e-7f ), tiny = set1( 1e-37f );
        for ( int c = 0; c < 3; ++c )
        {
            const F a = load( leaf.coord[0][c] ), b = load( leaf.coord[1][c] ), cc = load( leaf.coord[2][c] );
            const F l = min( min( a, b ), cc ), h = max( max( a, b ), cc );
            store( lo[c], sub( sub( l, mul( abs( l ), k ) ), tiny ) );
            store( hi[c], add( add( h, mul( abs( h ), k ) ), tiny ) );
        }
    }

    Box3f box( int i ) const { return Box3f( { lo[0][i], lo[1][i], lo[2][i] }, { hi[0][i], hi[1][i], hi[2][i] } ); }

    /// the mask of the boxes intersecting box i of other (as Box3f::intersects)
    int intersecting( const WideLeafTriBoxes & other, int i ) const
    {
        using namespace Simd4;
        M ok = andm( le( load( lo[0] ), set1( other.hi[0][i] ) ), le( set1( other.lo[0][i] ), load( hi[0] ) ) );
        ok = andm( ok, andm( le( load( lo[1] ), set1( other.hi[1][i] ) ), le( set1( other.lo[1][i] ), load( hi[1] ) ) ) );
        ok = andm( ok, andm( le( load( lo[2] ), set1( other.hi[2][i] ) ), le( set1( other.lo[2][i] ), load( hi[2] ) ) ) );
        return bits( ok );
    }
};

/// given as tris to forWideLeafTris, tests the boxes of two triangles by their intersection, all boxes of the other leaf at once
struct WideTriBoxesIntersect {};

/// calls onTris( i, j ) for the triangles (indices in leafTris()) of leaves s.a and s.b, whose boxes intersect:
/// triNode( i, iBox, bNode ) tests the box of triangle i (iBox contains aTree.leafBox( i ), see wideLeafTriBoxes) with the box of leaf s.b,
/// and tris( i, j, iBox, jBox ) the boxes of two triangles (jBox contains bTree.leafBox( j ));
/// in the traversal of one tree with itself (Self), a leaf paired with itself gives each pair of its triangles once, i < j;
/// stops and returns false as soon as onTris returns false
template<bool Self = false, typename TriNode, typename Tris, typename OnTris>
bool forWideLeafTris( const AABBTree & aTree, const AABBTree & bTree, const WideNodePair & s, TriNode && triNode, Tris && tris, OnTris && onTris )
{
    const auto & an = aTree.nodes()[s.a];
    const auto & bn = bTree.nodes()[s.b];
    assert( an.leaf() && bn.leaf() );
    const bool self = Self && s.a == s.b;
    const int aNum = an.numLeafTris();
    const int bNum = bn.numLeafTris();
    // the boxes of the triangles of both leaves, computed once
    const WideLeafTriBoxes aBoxes( aTree, s.a );
    const std::optional<WideLeafTriBoxes> bBoxesOpt = self ? std::nullopt : std::optional<WideLeafTriBoxes>( std::in_place, bTree, s.b );
    const WideLeafTriBoxes & bBoxes = self ? aBoxes : *bBoxesOpt;
    for ( int i = 0; i < aNum; ++i )
    {
        const auto iBox = aBoxes.box( i );
        if ( !self && !triNode( an.first + i, iBox, s.b ) )
            continue;
        if constexpr ( std::is_same_v<std::decay_t<Tris>, WideTriBoxesIntersect> )
        {
            int m = bBoxes.intersecting( aBoxes, i ) & ( ( 1 << bNum ) - 1 );
            if ( self )
                m &= ~0 << ( i + 1 );
            for ( ; m; m &= m - 1 )
                if ( const int j = Simd4::lowestBit( m ); !onTris( an.first + i, bn.first + j ) )
                    return false;
        }
        else
        {
            for ( int j = self ? i + 1 : 0; j < bNum; ++j )
                if ( tris( an.first + i, bn.first + j, iBox, bBoxes.box( j ) ) && !onTris( an.first + i, bn.first + j ) )
                    return false;
        }
    }
    return true;
}

/// sequentially subdivides the traversal from the given pairs on at most given number of levels to get subtasks for parallel processing:
/// they shall be not too many for this subdivision not to take too long, and not too few for enough parallelism later;
/// split( s, push ) returns false for a pair of leaves, which then becomes a subtask
template<typename Split>
Vector<WideNodePair> makeWideSubtasks( Vector<WideNodePair> subtasks, int levels, Split && split )
{
    Vector<WideNodePair> next, leafTasks;
    for ( int i = 0; i < levels && !subtasks.empty(); ++i )
    {
        for ( const auto & s : subtasks )
            if ( !split( s, [&next]( const WideNodePair & c ) { next.push_back( c ); } ) )
                leafTasks.push_back( s );
        subtasks.swap( next );
        next.clear();
    }
    subtasks.insert( subtasks.end(), leafTasks.begin(), leafTasks.end() );
    return subtasks;
}

/// the boxes after a transformation, each computed on the first request (from any thread) and kept for the next ones
template<typename I>
class LazyTransformedBoxes
{
public:
    LazyTransformedBoxes( size_t size, const AffineXf3f & xf ) : xf_( xf ), boxes_( size ), state_( size ) { }

    /// returns box i, which is src before the transformation
    Box3f get( I i, const Box3f & src ) const
    {
        auto & state = state_[i];
        if ( state.load( std::memory_order_acquire ) == cReady )
            return boxes_[i];
        const auto res = transformed( src, xf_ );
        // only the thread marking the box first stores it; the others meanwhile use their own copies
        std::uint8_t expected = cEmpty;
        if ( state.compare_exchange_strong( expected, cWriting, std::memory_order_relaxed ) )
        {
            boxes_[i] = res;
            state.store( cReady, std::memory_order_release );
        }
        return res;
    }

private:
    static constexpr std::uint8_t cEmpty = 0, cWriting = 1, cReady = 2;
    const AffineXf3f & xf_;
    mutable Buffer<Box3f, I> boxes_;
    mutable Vector<std::atomic<std::uint8_t>, I> state_;
};

/// visits depth-first the nodes with boxes passing nodeOk( box ), children in their order, and calls onTri( i )
/// for the triangles (indices in leafTris()) of the visited leaves with boxes passing triOk( box );
/// stops as soon as onTri returns Processing::Stop
template<typename NodeOk, typename TriOk, typename OnTri>
void forWideTris( const AABBTree & tree, NodeOk && nodeOk, TriOk && triOk, OnTri && onTri )
{
    const auto & nodes = tree.nodes();
    if ( nodes.empty() || !nodeOk( tree.getBoundingBox() ) )
        return;
    // every level adds at most MaxChildren - 1 nodes, and the tree of 2^31 triangles has 16 levels
    InplaceStack<NoInitNodeId, 64> subtasks;
    subtasks.push( NodeId( 0 ) );
    while ( !subtasks.empty() )
    {
        const auto & node = nodes[subtasks.top()];
        subtasks.pop();
        if ( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
                if ( triOk( tree.leafBox( i ) ) && onTri( i ) == Processing::Stop )
                    return;
            continue;
        }
        for ( int i = node.count - 1; i >= 0; --i )
            if ( const NodeId c( node.first + i ); nodeOk( tree.box( c ) ) )
                subtasks.push( c );
    }
}

} // namespace MR
