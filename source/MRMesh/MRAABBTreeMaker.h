#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"
#include "MRFunctional.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

template<typename T>
struct BoxedLeaf
{
    typename T::LeafId leafId;
    typename T::BoxT box;

    BoxedLeaf() noexcept = default;
    explicit BoxedLeaf( NoInit ) noexcept : leafId( noInit ), box( noInit ) { }
};

/// returns the number of nodes in the binary tree with given number of leaves
inline int getNumNodes( int numLeaves )
{
    assert( numLeaves >  0 );
    return 2 * numLeaves - 1;
}

template<typename T>
using OnLeafCallback = FunctionRef<typename T::LeafId( typename T::LeafId id, int pos )>;

/// \param onLeaf optional, called in parallel for every leaf with its id and position (AABBTreeNode::leafPos), returns the id to store in the leaf
template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( Buffer<BoxedLeaf<T>> boxedLeaves, const OnLeafCallback<T> * onLeaf = nullptr );

/// \}

} // namespace MR
