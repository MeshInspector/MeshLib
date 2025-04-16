#include "MRMeshTopologyDiff.h"
#include "MRTimer.h"
#include "MRHeapBytes.h"

namespace MR
{

MeshTopologyDiff::MeshTopologyDiff( const MeshTopology & from, const MeshTopology & to )
{
    MR_TIMER;

    toEdgesSize_ = to.edges_.size();
    for ( EdgeId e{0}; e < toEdgesSize_; ++e )
    {
        if ( e >= from.edges_.size() || from.edges_[e] != to.edges_[e] )
            changedEdges_[e] = to.edges_[e];
    }
}

void MeshTopologyDiff::applyAndSwap( MeshTopology & m )
{
    MR_TIMER;

    auto mEdgesSize = m.edges_.size();
    // remember edges_ being deleted from m
    for ( EdgeId e{toEdgesSize_}; e < mEdgesSize; ++e )
    {
        changedEdges_[e] = m.edges_[e];
    }
    m.edges_.resize( toEdgesSize_ );
    // swap common edges_ and delete edges_ for vertices missing in original m (that will be next target)
    for ( auto it = changedEdges_.begin(); it != changedEdges_.end(); )
    {
        auto e = it->first;
        auto & pos = it->second;
        if ( e < toEdgesSize_ )
        {
            std::swap( pos, m.edges_[e] );
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
