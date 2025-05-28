#include "MRVoxelsVolume.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelMinMax.h"

namespace MR
{

Expected<SimpleVolumeMinMax> functionVolumeToSimpleVolume( const FunctionVolume& volume, const ProgressCallback& cb )
{
    MR_TIMER;
    SimpleVolumeMinMax res;
    res.voxelSize = volume.voxelSize;
    res.dims = volume.dims;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    if ( !ParallelFor( 0_vox, indexer.endId(), [&]( VoxelId i )
    {
        res.data[i] = volume.data( indexer.toPos( i ) );
    }, cb ) )
        return unexpectedOperationCanceled();

    std::tie( res.min, res.max ) = parallelMinMax( res.data );
    return res;
}

} //namespace MR
