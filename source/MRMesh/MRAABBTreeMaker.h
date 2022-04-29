#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

template<typename T>
struct BoxedLeaf
{
    typename T::LeafId leafId;
    typename T::BoxT box;
};

/// returns the number of nodes in the binary tree with given number of leaves
inline int getNumNodes( int numLeaves )
{
    assert( numLeaves >  0 );
    return 2 * numLeaves - 1;
}

template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( std::vector<BoxedLeaf<T>> boxedLeaves );

/// \}

} // namespace MR
