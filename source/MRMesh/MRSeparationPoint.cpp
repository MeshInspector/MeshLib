#include "MRSeparationPoint.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

SeparationPointStorage::SeparationPointStorage( size_t blockCount, size_t blockSize )
    : blockSize_( blockSize )
    , hmaps_( blockCount )
{
}

void SeparationPointStorage::shiftVertIds( const std::function<int(size_t)> & getVertIndexShiftForVoxelId )
{
    MR_TIMER
    ParallelFor( size_t( 0 ), hmaps_.size(), [&] ( size_t hi )
    {
        for ( auto& [ind, set] : hmaps_[hi] )
        {
            auto vertShift = getVertIndexShiftForVoxelId( ind );
            for ( auto& sepPoint : set )
                if ( sepPoint )
                    sepPoint.vid += vertShift;
        }
    } );
}

void SeparationPointStorage::getPoints( VertCoords & points ) const
{
    MR_TIMER
    ParallelFor( size_t( 0 ), hmaps_.size(), [&] ( size_t hi )
    {
        for ( auto& [_, set] : hmaps_[hi] )
        {
            for ( int i = int( NeighborDir::X ); i < int( NeighborDir::Count ); ++i )
                if ( set[i].vid < points.size() )
                    points[set[i].vid] = set[i].position;
        }
    } );
}

} //namespace MR
