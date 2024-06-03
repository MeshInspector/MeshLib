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

AABBTreeObjects::AABBTreeObjects( const Vector<MeshOrPointsXf, ObjId> & objs )
{
    MR_TIMER
    using BoxedObj = BoxedLeaf<Traits>;
    Buffer<BoxedObj> boxedObjs( objs.size() );
    toWorld_.resize( objs.size() );
    toLocal_.resize( objs.size() );

    for ( ObjId oi(0); oi < objs.size(); ++oi )
    {
        boxedObjs[oi].leafId = oi;
        boxedObjs[oi].box = transformed( objs[oi].obj.getObjBoundingBox(), objs[oi].xf );
        toWorld_[oi] = objs[oi].xf;
        toLocal_[oi] = objs[oi].xf.inverse();
    }
    nodes_ = makeAABBTreeNodeVec( std::move( boxedObjs ) );
}

} //namespace MR
