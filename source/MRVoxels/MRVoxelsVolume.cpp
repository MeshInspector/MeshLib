#include "MRVoxelsVolume.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelFor.h"

namespace MR
{

Expected<SimpleVolumeMinMax> functionVolumeToSimpleVolume( const FunctionVolume& volume, const ProgressCallback& cb )
{
    MR_TIMER
    SimpleVolumeMinMax res;
    res.voxelSize = volume.voxelSize;
    res.dims = volume.dims;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    if ( !ParallelFor( size_t( 0 ), indexer.size(), [&]( size_t i )
    {
        res.data[i] = volume.data( indexer.toPos( VoxelId( i ) ) );
    }, cb ) )
        return unexpectedOperationCanceled();

    std::tie( res.min, res.max ) = parallelMinMax( res.data );
    return res;
}

}