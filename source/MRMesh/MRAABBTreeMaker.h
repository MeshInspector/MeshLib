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

/// called in parallel for every leaf with its index in the order of the tree and its index in the given boxed leaves
using OnLeafCallback = FunctionRef<void( size_t leafIndex, size_t boxedLeafIndex )>;

/// constructs the nodes of the tree with given leaves; the children of the nodes reference the leaves by their indices in the order of the tree,
/// which the caller receives in onLeaf to fill its leaves
template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( const Vector<BoxedLeaf<T>> & boxedLeaves, OnLeafCallback onLeaf );

/// \}

} // namespace MR
