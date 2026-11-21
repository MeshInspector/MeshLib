#include "MRAABBTreeMaker.h"
#include "MRAABBTreeNode.h"
#include "MRAABBTreeBase.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRSelectByKey.hpp"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace MR
{

namespace
{

/// splits each node at the median of the leaf box centers along the largest extent of these centers;
/// node boxes are united from the children's boxes
template<typename T>
class AABBTreeMaker
{
public:
    using Node = AABBTreeNode<T>;
    using NodeVec = Vector<Node, NodeId>;
    using BoxT = typename T::BoxT;
    using V = decltype( BoxT{}.min );

    AABBTreeMaker( const Vector<BoxedLeaf<T>> & boxedLeaves, const OnLeafCallback & onLeaf )
        : boxedLeaves_( boxedLeaves ), onLeaf_( onLeaf ) { }
    NodeVec construct();

private:
    /// sorted instead of BoxedLeaf: smaller and compared by one coordinate
    struct Center
    {
        int i; ///< in boxedLeaves_
        V c; ///< doubled center of the box

        Center( int i, const V & c ) noexcept : i( i ), c( c ) { }
        explicit Center( NoInit ) noexcept : c( noInit ) { }
    };

    const Vector<BoxedLeaf<T>> & boxedLeaves_;
    const OnLeafCallback & onLeaf_; // a copy would reference the source FunctionRef via its template constructor
    Buffer<Center> centers_;
    Buffer<Center> scratch_; // for the splits of big ranges
    // a range of at least so many centers is split by chunks in parallel; on one thread too, for the same places of equal centers
    static constexpr int cParallelSplitNum = 1 << 17;
    NodeVec nodes_;

    int largestCentersAxis_( int first, int num ) const;
    // [first, first + num / 2) get the smaller centers along axis, placed the same with any standard library and number of threads
    void split_( int first, int num, int axis );
    const BoxT & childBox_( NodeId c ) const { return Node::isLeaf( c ) ? boxedLeaves_[centers_[Node::leafIndex( c )].i].box : nodes_[c].box; }
    // fills node n with the subtree of at least two leaves centers_[first, first + num);
    // its children that are nodes are placed at next and next + 1, followed by the other nodes of the left subtree, then of the right one
    void make_( NodeId n, int first, int num, NodeId next );
    void makeChild_( NodeId c, int first, int num, NodeId next );
};

template<typename T>
int AABBTreeMaker<T>::largestCentersAxis_( int first, int num ) const
{
    BoxT box;
    if ( num >= cParallelSplitNum )
    {
        box = tbb::parallel_reduce( tbb::blocked_range<int>( first, first + num, 1 << 14 ), BoxT(),
            [&]( const tbb::blocked_range<int> & r, BoxT b )
            {
                for ( int i = r.begin(); i < r.end(); ++i )
                    b.include( centers_[i].c );
                return b;
            },
            []( BoxT a, const BoxT & b ) { a.include( b ); return a; } );
    }
    else
    {
        for ( int i = 0; i < num; ++i )
            box.include( centers_[first + i].c );
    }
    const auto sz = box.size();
    int axis = 0;
    for ( int d = 1; d < BoxT::elements; ++d )
        if ( sz[d] > sz[axis] )
            axis = d;
    return axis;
}

template<typename T>
void AABBTreeMaker<T>::split_( int first, int num, int axis )
{
    const auto key = [axis]( const Center & c ) { return c.c[axis]; };
    Center * const data = centers_.data() + first;
    const int mid = num / 2;
    if ( num < cParallelSplitNum )
    {
        selectByKey( data, data + mid, data + num, key );
        return;
    }

    // a big range is partitioned in parallel in three parts by two sample coordinates around the median,
    // and only the small middle part with the median is given to selectByKey
    constexpr int cNumSamples = 1024, cHalfGap = 64;
    using Coord = decltype( V{}[0] + 0 );
    Coord samples[cNumSamples];
    for ( int k = 0; k < cNumSamples; ++k )
        samples[k] = key( data[ std::int64_t( k ) * num / cNumSamples ] );
    std::sort( samples, samples + cNumSamples );
    const Coord lo = samples[cNumSamples / 2 - cHalfGap];
    const Coord hi = samples[cNumSamples / 2 + cHalfGap];
    const auto part = [&]( const Center & c ) { return key( c ) < lo ? 0 : ( key( c ) < hi ? 1 : 2 ); };

    constexpr int cChunk = 1 << 13;
    const int numChunks = ( num + cChunk - 1 ) / cChunk;
    struct Counts { int n[3]; };
    std::vector<Counts> counts( numChunks );
    ParallelFor( 0, numChunks, [&]( int ch )
    {
        Counts cnt{};
        for ( int i = ch * cChunk, e = std::min( num, i + cChunk ); i < e; ++i )
            ++cnt.n[part( data[i] )];
        counts[ch] = cnt;
    } );
    Counts total{};
    for ( const auto & c : counts )
        for ( int k = 0; k < 3; ++k )
            total.n[k] += c.n[k];
    if ( mid < total.n[0] || mid >= total.n[0] + total.n[1] )
    {
        // the median is not between the samples
        selectByKey( data, data + mid, data + num, key );
        return;
    }
    Counts pos{ { 0, total.n[0], total.n[0] + total.n[1] } };
    for ( auto & c : counts )
    {
        const auto my = pos;
        for ( int k = 0; k < 3; ++k )
            pos.n[k] += c.n[k];
        c = my;
    }
    Center * const tmp = scratch_.data() + first;
    ParallelFor( 0, numChunks, [&]( int ch )
    {
        auto p = counts[ch];
        for ( int i = ch * cChunk, e = std::min( num, i + cChunk ); i < e; ++i )
            tmp[ p.n[part( data[i] )]++ ] = data[i];
    } );
    ParallelFor( 0, numChunks, [&]( int ch )
    {
        const int i = ch * cChunk, e = std::min( num, i + cChunk );
        std::copy( tmp + i, tmp + e, data + i );
    } );
    selectByKey( data + total.n[0], data + mid, data + total.n[0] + total.n[1], key );
}

template<typename T>
void AABBTreeMaker<T>::makeChild_( NodeId c, int first, int num, NodeId next )
{
    if ( num == 1 )
        onLeaf_( first, centers_[first].i );
    else
        make_( c, first, num, next );
}

template<typename T>
void AABBTreeMaker<T>::make_( NodeId n, int first, int num, NodeId next )
{
    assert( num >= 2 );
    const int numL = num / 2;
    const int numR = num - numL;
    split_( first, num, largestCentersAxis_( first, num ) );

    auto & node = nodes_[n];
    NodeId slot = next;
    node.l = numL == 1 ? Node::leafChild( first ) : slot++;
    node.r = numR == 1 ? Node::leafChild( first + numL ) : slot++;
    const NodeId nextL = slot;
    const NodeId nextR = numL == 1 ? nextL : nextL + getNumNodes( numL ) - 1;
    if ( num < 4096 )
    {
        makeChild_( node.l, first, numL, nextL );
        makeChild_( node.r, first + numL, numR, nextR );
    }
    else
    {
        tbb::task_group group;
        group.run( [&] { makeChild_( node.r, first + numL, numR, nextR ); } );
        makeChild_( node.l, first, numL, nextL );
        group.wait();
    }
    auto box = childBox_( node.l );
    box.include( childBox_( node.r ) );
    nodes_[n].box = box;
}

} // anonymous namespace

template<typename T>
auto AABBTreeMaker<T>::construct() -> NodeVec
{
    MR_TIMER;

    const auto numLeaves = (int)boxedLeaves_.size();
    assert( numLeaves > 0 );
    centers_.resize( numLeaves );
    if ( numLeaves >= cParallelSplitNum )
        scratch_ = Buffer<Center>( numLeaves );
    ParallelFor( 0, numLeaves, [&]( int i )
    {
        const auto & box = boxedLeaves_[i].box;
        centers_[i] = Center( i, box.min + box.max );
    } );

    nodes_.resizeNoInit( getNumNodes( numLeaves ) );
    makeChild_( NodeId{ 0 }, 0, numLeaves, NodeId{ 1 } );

    return std::move( nodes_ );
}

template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( const Vector<BoxedLeaf<T>> & boxedLeaves, OnLeafCallback onLeaf )
{
    return AABBTreeMaker<T>( boxedLeaves, onLeaf ).construct();
}

template<typename T>
void AABBTreeBase<T>::build_( Vector<BoxedLeaf<T>> leaves )
{
    nodes_.clear();
    leaves_.clear();
    if ( leaves.empty() )
        return;
    leaves_.resizeNoInit( leaves.size() );
    nodes_ = makeAABBTreeNodeVec( leaves, [&]( size_t leafIndex, size_t boxedLeafIndex ) { leaves_[leafIndex] = leaves[boxedLeafIndex]; } );
}

} //namespace MR
