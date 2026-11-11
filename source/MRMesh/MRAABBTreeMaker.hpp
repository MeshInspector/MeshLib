#include "MRAABBTreeMaker.h"
#include "MRAABBTreeNode.h"
#include "MRAABBTreeBase.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
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
    using Node = typename AABBTreeNodeType<T>::type;
    using NodeVec = Vector<Node, NodeId>;
    using BoxT = typename T::BoxT;
    using V = decltype( BoxT{}.min );

    AABBTreeMaker( const Vector<BoxedLeaf<T>> & boxedLeaves, const OnLeafCallback & onLeaf, BoxT * nodeBoxes )
        : boxedLeaves_( boxedLeaves ), onLeaf_( onLeaf ), nodeBoxes_( nodeBoxes ) { }
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
    Buffer<Center> scratch_; // for parallel splits of big ranges
    static constexpr int cParallelSplitNum = 1 << 17;
    bool parallelSplits_ = false; // on one thread the parallel split only adds passes over the range
    NodeVec nodes_;
    BoxT * nodeBoxes_ = nullptr; // for the nodes without boxes

    BoxT & nodeBox_( NodeId n ) { if constexpr ( nodesHaveBoxes<T> ) return nodes_[n].box; else return nodeBoxes_[int( n )]; }
    const BoxT & nodeBox_( NodeId n ) const { if constexpr ( nodesHaveBoxes<T> ) return nodes_[n].box; else return nodeBoxes_[int( n )]; }
    int largestCentersAxis_( int first, int num ) const;
    // [first, first + num / 2) get the smaller centers along axis; ties are ordered by index to make the tree independent of std::nth_element
    void split_( int first, int num, int axis );
    const BoxT & childBox_( NodeId c ) const { return Node::isLeaf( c ) ? boxedLeaves_[centers_[Node::leafIndex( c )].i].box : nodeBox_( c ); }
    // fills node n with the subtree of at least two leaves centers_[first, first + num);
    // its children that are nodes are placed at next and next + 1, followed by the other nodes of the left subtree, then of the right one
    void make_( NodeId n, int first, int num, NodeId next );
    void makeChild_( NodeId c, int first, int num, NodeId next );
};

template<typename T>
int AABBTreeMaker<T>::largestCentersAxis_( int first, int num ) const
{
    BoxT box;
    if ( parallelSplits_ && num >= cParallelSplitNum )
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
    const auto less = [axis]( const Center & a, const Center & b ) { return a.c[axis] < b.c[axis] || ( a.c[axis] == b.c[axis] && a.i < b.i ); };
    Center * const data = centers_.data() + first;
    const int mid = num / 2;
    if ( !parallelSplits_ || num < cParallelSplitNum )
    {
        std::nth_element( data, data + mid, data + num, less );
        return;
    }

    // a big range is partitioned in parallel in three parts by two samples around the median,
    // and only the small middle part with the median is given to std::nth_element
    constexpr int cNumSamples = 1024, cHalfGap = 64;
    Buffer<Center> samples( cNumSamples );
    for ( int k = 0; k < cNumSamples; ++k )
        samples[k] = data[ std::int64_t( k ) * num / cNumSamples ];
    std::sort( samples.data(), samples.data() + cNumSamples, less );
    const Center lo = samples[cNumSamples / 2 - cHalfGap];
    const Center hi = samples[cNumSamples / 2 + cHalfGap];
    const auto part = [&]( const Center & c ) { return less( c, lo ) ? 0 : ( less( c, hi ) ? 1 : 2 ); };

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
        std::nth_element( data, data + mid, data + num, less );
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
    std::nth_element( data + total.n[0], data + mid, data + total.n[0] + total.n[1], less );
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
    nodeBox_( n ) = box;
}

} // anonymous namespace

template<typename T>
auto AABBTreeMaker<T>::construct() -> NodeVec
{
    MR_TIMER;

    const auto numLeaves = (int)boxedLeaves_.size();
    assert( numLeaves > 0 );
    assert( nodesHaveBoxes<T> || nodeBoxes_ || numLeaves == 1 );
    centers_.resize( numLeaves );
    parallelSplits_ = numLeaves >= cParallelSplitNum
        && tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) > 1
        && tbb::this_task_arena::max_concurrency() > 1;
    if ( parallelSplits_ )
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
AABBTreeNodeVec<T> makeAABBTreeNodeVec( const Vector<BoxedLeaf<T>> & boxedLeaves, OnLeafCallback onLeaf, typename T::BoxT * nodeBoxes )
{
    return AABBTreeMaker<T>( boxedLeaves, onLeaf, nodeBoxes ).construct();
}

template<typename T>
void AABBTreeBase<T>::build_( Vector<BoxedLeaf<T>> leaves ) requires leavesHaveBoxes
{
    nodes_.clear();
    leaves_.clear();
    if ( leaves.empty() )
        return;
    leaves_.resizeNoInit( leaves.size() );
    nodes_ = makeAABBTreeNodeVec( leaves, [&]( size_t leafIndex, size_t boxedLeafIndex ) { leaves_[leafIndex] = leaves[boxedLeafIndex]; } );
}

} //namespace MR
