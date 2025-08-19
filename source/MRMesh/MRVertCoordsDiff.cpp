#include "MRVertCoordsDiff.h"
#include "MRVector.h"
#include "MRVector3.h"
#include "MRTimer.h"
#include "MRHeapBytes.h"
#include "MRId.h"

namespace MR
{

VertCoordsDiff::VertCoordsDiff( const VertCoords & from, const VertCoords & to )
{
    MR_TIMER;

    toPointsSize_ = to.size();
    for ( VertId v{0}; v < toPointsSize_; ++v )
    {
        if ( v >= from.size() || from[v] != to[v] )
            changedPoints_[v] = to[v];
    }
}

void VertCoordsDiff::applyAndSwap( VertCoords & m )
{
    MR_TIMER;

    auto mPointsSize = m.size();
    // remember points being deleted from m
    for ( VertId v{toPointsSize_}; v < mPointsSize; ++v )
    {
        changedPoints_[v] = m[v];
    }
    m.resize( toPointsSize_ );
    // swap common points and delete points for vertices missing in original m (that will be next target)
    for ( auto it = changedPoints_.begin(); it != changedPoints_.end(); )
    {
        auto v = it->first;
        auto & pos = it->second;
        if ( v < toPointsSize_ )
        {
            std::swap( pos, m[v] );
            if ( v >= mPointsSize )
            {
                it = changedPoints_.erase( it );
                continue;
            }
        }
        ++it;
    }
    toPointsSize_ = mPointsSize;
}

size_t VertCoordsDiff::heapBytes() const
{
    return MR::heapBytes( changedPoints_ );
}

} // namespace MR
