#pragma once

#include "MRAABBTreeWide.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include "MRBuffer.h"
#include "MRVector.h"
#include <atomic>
#include <cstdint>

namespace MR
{

/// a pair of nodes of two AABBTreeWide (or of one tree) in their simultaneous traversal
struct WideNodePair
{
    NodeId a, b;
};

/// pushes the pairs of each child of the bigger non-leaf node with the other node, if intersect( aNode, bNode ) of their boxes;
/// returns false and pushes nothing if both nodes are leaves
template<typename Intersect, typename Push>
bool splitWidePair( const AABBTreeWide & aTree, const AABBTreeWide & bTree, const WideNodePair & s, Intersect && intersect, Push && push )
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
bool splitSelfWidePair( const AABBTreeWide & tree, const WideNodePair & s, Intersect && intersect, Push && push )
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
bool forWideLeafTris( const AABBTreeWide & aTree, const AABBTreeWide & bTree, const WideNodePair & s, TriNode && triNode, Tris && tris, OnTris && onTris )
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

/// returns the nodes of the tree having at least one triangle of the region
inline NodeBitSet getWideNodesWithFaces( const AABBTreeWide & tree, const FaceBitSet & region )
{
    NodeBitSet res( tree.nodes().size() );
    // children follow their parent
    for ( int n = (int)tree.nodes().size() - 1; n >= 0; --n )
    {
        const auto & node = tree.nodes()[NodeId( n )];
        bool any = false;
        if ( node.leaf() )
        {
            for ( int i = 0; i < node.numLeafTris() && !any; ++i )
                any = region.test( tree.leafTris()[node.first + i].face );
        }
        else
        {
            for ( int i = 0; i < node.count && !any; ++i )
                any = res.test( NodeId( node.first + i ) );
        }
        if ( any )
            res.set( NodeId( n ) );
    }
    return res;
}

} // namespace MR
