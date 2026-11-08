#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"
#include "MRFunctional.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/// returns the number of not-leaf nodes in the binary tree with given number of leaves
inline int getNumNodes( int numLeaves )
{
    assert( numLeaves >  0 );
    return numLeaves - 1;
}

/// called in parallel for every leaf with its final index in the reordered leaves
using OnLeafCallback = FunctionRef<void( size_t leafIndex )>;

/// constructs the nodes of the tree with given leaves, and reorders the leaves in the order of the tree;
/// the children of the nodes reference the leaves by their indices after reordering
template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( Vector<BoxedLeaf<T>> & boxedLeaves, const OnLeafCallback * onLeaf = nullptr );

/// \}

} // namespace MR
