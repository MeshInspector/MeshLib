#include "MRAABBTreeMaker.h"
#include "MRAABBTreeNode.h"
#include "MRAABBTreeBase.h"
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

    AABBTreeMaker( const Vector<BoxedLeaf<T>> & boxedLeaves, OnLeafCallback onLeaf ) : boxedLeaves_( boxedLeaves ), onLeaf_( onLeaf ) { }
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
    OnLeafCallback onLeaf_;
    Buffer<Center> centers_;
    NodeVec nodes_;

    int largestCentersAxis_( int first, int num ) const;
    const BoxT & childBox_( NodeId c ) const { return Node::isLeaf( c ) ? boxedLeaves_[centers_[Node::leafIndex( c )].i].box : nodes_[c].box; }
    // fills node n with the subtree of at least two leaves centers_[first, first + num)
    void make_( NodeId n, int first, int num );
    void makeChild_( NodeId c, int first, int num );
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
void AABBTreeMaker<T>::makeChild_( NodeId c, int first, int num )
{
    if ( num == 1 )
        onLeaf_( first, centers_[first].i );
    else
        make_( c, first, num );
}

template<typename T>
void AABBTreeMaker<T>::make_( NodeId n, int first, int num )
{
    assert( num >= 2 );
    // ties are ordered by index to make the tree independent of std::nth_element
    const int axis = largestCentersAxis_( first, num );
    const int numL = num / 2;
    const int numR = num - numL;
    std::nth_element( centers_.data() + first, centers_.data() + first + numL, centers_.data() + first + num,
        [axis]( const Center & a, const Center & b ) { return a.c[axis] < b.c[axis] || ( a.c[axis] == b.c[axis] && a.i < b.i ); } );

    auto & node = nodes_[n];
    node.l = numL == 1 ? Node::leafChild( first ) : n + 1;
    node.r = numR == 1 ? Node::leafChild( first + numL ) : n + 1 + getNumNodes( numL );
    if ( num < 4096 )
    {
        makeChild_( node.l, first, numL );
        makeChild_( node.r, first + numL, numR );
    }
    else
    {
        tbb::task_group group;
        group.run( [&] { makeChild_( node.r, first + numL, numR ); } );
        makeChild_( node.l, first, numL );
        group.wait();
    }
    auto box = childBox_( node.l );
    box.include( childBox_( node.r ) );
    node.box = box;
}

} // anonymous namespace

template<typename T>
auto AABBTreeMaker<T>::construct() -> NodeVec
{
    MR_TIMER;

    const auto numLeaves = (int)boxedLeaves_.size();
    assert( numLeaves > 0 );
    centers_.resize( numLeaves );
    ParallelFor( 0, numLeaves, [&]( int i )
    {
        const auto & box = boxedLeaves_[i].box;
        centers_[i] = Center( i, box.min + box.max );
    } );

    nodes_.resizeNoInit( getNumNodes( numLeaves ) );
    makeChild_( NodeId{ 0 }, 0, numLeaves );

    return std::move( nodes_ );
}

template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( const Vector<BoxedLeaf<T>> & boxedLeaves, OnLeafCallback onLeaf )
{
    return AABBTreeMaker<T>( boxedLeaves, onLeaf ).construct();
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
