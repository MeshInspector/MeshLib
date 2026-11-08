#include "MRAABBTreeMaker.h"
#include "MRAABBTreeNode.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <algorithm>

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

    NodeVec construct( Buffer<BoxedLeaf<T>> boxedLeaves, const OnLeafCallback<T> * onLeaf );

private:
    /// sorted instead of BoxedLeaf: smaller and compared by one coordinate
    struct Center
    {
        int i; ///< in boxedLeaves_
        V c; ///< doubled center of the box

        Center( int i, const V & c ) noexcept : i( i ), c( c ) { }
        explicit Center( NoInit ) noexcept : c( noInit ) { }
    };

    Buffer<BoxedLeaf<T>> boxedLeaves_;
    Buffer<Center> centers_;
    const OnLeafCallback<T> * onLeaf_ = nullptr;
    NodeVec nodes_;

    int largestCentersAxis_( int first, int num ) const;
    // fills node n with the subtree of leaves centers_[first, first + num);
    // its two children are placed at (next, next + 1), followed by the other nodes of the left subtree, then of the right one
    void make_( NodeId n, int first, int num, NodeId next );
};

template<typename T>
int AABBTreeMaker<T>::largestCentersAxis_( int first, int num ) const
{
    BoxT box;
    for ( int i = 0; i < num; ++i )
        box.include( centers_[first + i].c );
    const auto sz = box.size();
    int axis = 0;
    for ( int d = 1; d < BoxT::elements; ++d )
        if ( sz[d] > sz[axis] )
            axis = d;
    return axis;
}

template<typename T>
void AABBTreeMaker<T>::make_( NodeId n, int first, int num, NodeId next )
{
    auto & node = nodes_[n];
    if ( num == 1 )
    {
        const auto & bl = boxedLeaves_[centers_[first].i];
        node.setLeafId( onLeaf_ ? ( *onLeaf_ )( bl.leafId, first ) : bl.leafId, first );
        node.box = bl.box;
        return;
    }

    // ties are ordered by index to make the tree independent of std::nth_element
    const int axis = largestCentersAxis_( first, num );
    const int numL = num / 2;
    std::nth_element( centers_.data() + first, centers_.data() + first + numL, centers_.data() + first + num,
        [axis]( const Center & a, const Center & b ) { return a.c[axis] < b.c[axis] || ( a.c[axis] == b.c[axis] && a.i < b.i ); } );

    node.l = next;
    node.r = next + 1;
    const NodeId nextL = next + 2;
    const NodeId nextR = nextL + getNumNodes( numL ) - 1;
    if ( num < 4096 )
    {
        make_( node.l, first, numL, nextL );
        make_( node.r, first + numL, num - numL, nextR );
    }
    else
    {
        tbb::task_group group;
        group.run( [&] { make_( node.r, first + numL, num - numL, nextR ); } );
        make_( node.l, first, numL, nextL );
        group.wait();
    }
    auto box = nodes_[node.l].box;
    box.include( nodes_[node.r].box );
    node.box = box;
}

} // anonymous namespace

template<typename T>
auto AABBTreeMaker<T>::construct( Buffer<BoxedLeaf<T>> boxedLeaves, const OnLeafCallback<T> * onLeaf ) -> NodeVec
{
    MR_TIMER;

    boxedLeaves_ = std::move( boxedLeaves );
    onLeaf_ = onLeaf;
    const auto numLeaves = (int)boxedLeaves_.size();
    centers_.resize( numLeaves );
    ParallelFor( 0, numLeaves, [&]( int i )
    {
        const auto & box = boxedLeaves_[i].box;
        centers_[i] = Center( i, box.min + box.max );
    } );

    nodes_.resizeNoInit( getNumNodes( numLeaves ) );
    make_( NodeId{ 0 }, 0, numLeaves, NodeId{ 1 } );

    return std::move( nodes_ );
}

template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( Buffer<BoxedLeaf<T>> boxedLeaves, const OnLeafCallback<T> * onLeaf )
{
    return AABBTreeMaker<T>().construct( std::move( boxedLeaves ), onLeaf );
}

} //namespace MR
