#include "MRAABBTreeObjects.h"
#include "MRAABBTreeBase.hpp"
#include "MRBuffer.h"

namespace MR
{

template auto AABBTreeBase<ObjTreeTraits>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<ObjTreeTraits>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<ObjTreeTraits>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<ObjTreeTraits>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<ObjTreeTraits>::getLeafOrderAndReset( LeafBMap & leafMap );

} //namespace MR
