#pragma once

#include "MRAABBTree.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include "MRBuffer.h"
#include "MREnums.h"
#include "MRInplaceStack.h"
#include "MRVector.h"
#include <atomic>
#include <cstdint>

namespace MR
{

/// a pair of nodes of two AABBTree (or of one tree) in their simultaneous traversal
struct WideNodePair
{
    NodeId a, b;
};

/// pushes the pairs of each child of the bigger non-leaf node with the other node, if intersect( aNode, bNode ) of their boxes;
/// returns false and pushes nothing if both nodes are leaves
template<typename Intersect, typename Push>
bool splitWidePair( const AABBTree & aTree, const AABBTree & bTree, const WideNodePair & s, Intersect && intersect, Push && push )
{
    const auto & an = aTree.nodes()[s.a];
    const auto & bn = bTree.nodes()[s.b];
    if ( !an.leaf() && ( bn.leaf() || an.box.volume() >= bn.box.volume() ) )
    {
        for ( int i = 0; i < an.count; ++i )
            if ( const NodeId c( an.first + i ); intersect( c, s.b ) )
                push( WideNodePair{ c, s.b } );
        return true;
    }
    if ( !bn.leaf() )
    {
        for ( int i = 0; i < bn.count; ++i )
            if ( const NodeId c( bn.first + i ); intersect( s.a, c ) )
                push( WideNodePair{ s.a, c } );
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
        for ( int j = i + 1; j < n.count; ++j )
            if ( const NodeId cj( n.first + j ); intersect( ci, cj ) )
                push( WideNodePair{ ci, cj } );
    }
    return true;
}

/// calls onTris( i, j ) for the triangles (indices in leafTris()) of leaves s.a and s.b, whose boxes intersect:
/// triNode( i, bNode ) tests the box of triangle i with the box of leaf s.b, and tris( i, j ) the boxes of two triangles;
/// in the traversal of one tree with itself (Self), a leaf paired with itself gives each pair of its triangles once, i < j;
/// stops and returns false as soon as onTris returns false
template<bool Self = false, typename TriNode, typename Tris, typename OnTris>
bool forWideLeafTris( const AABBTree & aTree, const AABBTree & bTree, const WideNodePair & s, TriNode && triNode, Tris && tris, OnTris && onTris )
{
    const auto & an = aTree.nodes()[s.a];
    const auto & bn = bTree.nodes()[s.b];
    assert( an.leaf() && bn.leaf() );
    const bool self = Self && s.a == s.b;
    const int aEnd = an.first + an.numLeafTris();
    const int bEnd = bn.first + bn.numLeafTris();
    for ( int i = an.first; i < aEnd; ++i )
    {
        if ( !self && !triNode( i, s.b ) )
            continue;
        for ( int j = self ? i + 1 : bn.first; j < bEnd; ++j )
            if ( tris( i, j ) && !onTris( i, j ) )
                return false;
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
        return getOrMake( i, [&] { return transformed( src, xf_ ); } );
    }

    /// returns box i, which is make() if it was not requested before
    template<typename Make>
    Box3f getOrMake( I i, Make && make ) const
    {
        auto & state = state_[i];
        if ( state.load( std::memory_order_acquire ) == cReady )
            return boxes_[i];
        const Box3f res = make();
        // only the thread marking the box first stores it; the others meanwhile use their own copies
        std::uint8_t expected = cEmpty;
        if ( state.compare_exchange_strong( expected, cWriting, std::memory_order_relaxed ) )
        {
            boxes_[i] = res;
            state.store( cReady, std::memory_order_release );
        }
        return res;
    }

    const AffineXf3f & xf() const { return xf_; }

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
    if ( nodes.empty() || !nodeOk( nodes.front().box ) )
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
                if ( triOk( tree.leafBoxes()[i] ) && onTri( i ) == Processing::Stop )
                    return;
            continue;
        }
        for ( int i = node.count - 1; i >= 0; --i )
            if ( const NodeId c( node.first + i ); nodeOk( nodes[c].box ) )
                subtasks.push( c );
    }
}

} // namespace MR
