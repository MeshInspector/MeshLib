#include "MRMeshTopologyDiff.h"
#include "MRTimer.h"
#include "MRHeapBytes.h"

namespace MR
{

MeshTopologyDiff::MeshTopologyDiff( const MeshTopology & from, const MeshTopology & to )
{
    MR_TIMER;

    toEdgesSize_ = to.edgeSize();
    for ( EdgeId e{0}; e < toEdgesSize_; ++e )
    {
        if ( e >= from.edgeSize() || from.getHalfEdge_( e ) != to.getHalfEdge_( e ) )
            changedEdges_[e] = to.getHalfEdge_( e );
    }
}

void MeshTopologyDiff::applyAndSwap( MeshTopology & m )
{
    MR_TIMER;

    auto mEdgesSize = m.edgeSize();
    // remember edges_ being deleted from m
    for ( EdgeId e{toEdgesSize_}; e < mEdgesSize; ++e )
    {
        changedEdges_[e] = m.getHalfEdge_( e );
    }
    m.next_.resize( toEdgesSize_ );
    m.prev_.resize( toEdgesSize_ );
    m.org_.resize( toEdgesSize_ );
    m.left_.resize( toEdgesSize_ );
    // swap common edges_ and delete edges_ for vertices missing in original m (that will be next target)
    for ( auto it = changedEdges_.begin(); it != changedEdges_.end(); )
    {
        auto e = it->first;
        auto & pos = it->second;
        if ( e < toEdgesSize_ )
        {
            m.swapHalfEdge_( e, pos );
            if ( e >= mEdgesSize )
            {
                it = changedEdges_.erase( it );
                continue;
            }
        }
        ++it;
    }
    toEdgesSize_ = mEdgesSize;

    m.computeAllFromEdges_();
}

size_t MeshTopologyDiff::heapBytes() const
{
    return MR::heapBytes( changedEdges_ );
}
} // namespace MR
