#include "MRSeparationPoint.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

SeparationPointStorage::SeparationPointStorage( size_t blockCount, size_t blockSize )
    : blockSize_( blockSize )
    , blocks_( blockCount )
{
}

int SeparationPointStorage::makeUniqueVids()
{
    MR_TIMER
    std::vector<int> block2shift;
    block2shift.reserve( blocks_.size() );
    int lastShift = 0;
    for ( auto & b : blocks_ )
    {
        block2shift.push_back( lastShift );
        lastShift += b.nextVid;
    }

    ParallelFor( size_t( 0 ), blocks_.size(), [&] ( size_t bi )
    {
        const auto shift = block2shift[bi];
        for ( auto& [_, set] : blocks_[bi].smap )
        {
            for ( auto& sepPoint : set )
                if ( sepPoint )
                    sepPoint.vid += shift;
        }
    } );
    return lastShift;
}

void SeparationPointStorage::getPoints( VertCoords & points ) const
{
    MR_TIMER
    ParallelFor( size_t( 0 ), blocks_.size(), [&] ( size_t hi )
    {
        for ( auto& [_, set] : blocks_[hi].smap )
        {
            for ( int i = int( NeighborDir::X ); i < int( NeighborDir::Count ); ++i )
                if ( set[i].vid < points.size() )
                    points[set[i].vid] = set[i].position;
        }
    } );
}

} //namespace MR
