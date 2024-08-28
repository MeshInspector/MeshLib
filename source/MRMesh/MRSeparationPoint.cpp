#include "MRSeparationPoint.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

void SeparationPointStorage::resize( size_t blockCount, size_t blockSize )
{
    blockSize_ = blockSize;
    blocks_.resize( blockCount );
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

Triangulation SeparationPointStorage::getTriangulation( Vector<VoxelId, FaceId>* outVoxelPerFaceMap ) const
{
    MR_TIMER
    size_t totalTris = 0;
    for ( const auto & b : blocks_ )
        totalTris += b.tris.size();

    Triangulation res;
    res.reserve( totalTris );
    if ( outVoxelPerFaceMap )
    {
        outVoxelPerFaceMap->clear();
        outVoxelPerFaceMap->reserve( totalTris );
    }
    for ( const auto & b : blocks_ )
    {
        res.vec_.insert( end( res ), begin( b.tris ), end( b.tris ) );
        if ( outVoxelPerFaceMap )
            outVoxelPerFaceMap->vec_.insert( end( *outVoxelPerFaceMap ),
                begin( b.faceMap ), end( b.faceMap ) );
    }
    return res;
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
