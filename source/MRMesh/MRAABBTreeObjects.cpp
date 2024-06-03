#include "MRAABBTreeObjects.h"
#include "MRAABBTreeBase.hpp"
#include "MRAABBTreeMaker.hpp"
#include "MRBuffer.h"
#include "MRMeshOrPoints.h"

namespace MR
{

template auto AABBTreeBase<ObjTreeTraits>::getSubtrees( int minNum ) const -> std::vector<NodeId>;
template auto AABBTreeBase<ObjTreeTraits>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet;
template NodeBitSet AABBTreeBase<ObjTreeTraits>::getNodesFromLeaves( const LeafBitSet & leaves ) const;
template void AABBTreeBase<ObjTreeTraits>::getLeafOrder( LeafBMap & leafMap ) const;
template void AABBTreeBase<ObjTreeTraits>::getLeafOrderAndReset( LeafBMap & leafMap );

AABBTreeObjects::AABBTreeObjects( Vector<MeshOrPointsXf, ObjId> objs ) : objs_( std::move( objs ) )
{
    MR_TIMER
    using BoxedObj = BoxedLeaf<Traits>;
    Buffer<BoxedObj> boxedObjs( objs_.size() );
    toLocal_.resize( objs_.size() );

    for ( ObjId oi(0); oi < objs_.size(); ++oi )
    {
        boxedObjs[oi].leafId = oi;
        boxedObjs[oi].box = transformed( objs_[oi].obj.getObjBoundingBox(), objs_[oi].xf );
        toLocal_[oi] = objs_[oi].xf.inverse();
    }
    nodes_ = makeAABBTreeNodeVec( std::move( boxedObjs ) );
}

} //namespace MR
