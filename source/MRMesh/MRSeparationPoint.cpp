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
    VertId lastShift{ 0 };
    for ( auto & b : blocks_ )
    {
        b.shift = lastShift;
        lastShift += b.nextVid();
    }

    ParallelFor( size_t( 0 ), blocks_.size(), [&] ( size_t bi )
    {
        const auto shift = blocks_[bi].shift;
        for ( auto& [_, set] : blocks_[bi].smap )
        {
            for ( auto& sepPoint : set )
                if ( sepPoint )
                    sepPoint += shift;
        }
    } );
    return lastShift;
}

void SeparationPointStorage::getPoints( VertCoords & points ) const
{
    MR_TIMER
    ParallelFor( size_t( 0 ), blocks_.size(), [&] ( size_t bi )
    {
        VertId v = blocks_[bi].shift;
        for ( const auto & p : blocks_[bi].coords )
            points[v++] = p;
    } );
}

} //namespace MR
