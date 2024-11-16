#include "MRVoxelFilter.h"

#include <MRVoxels/MRVoxelsVolume.h>
#include <MRVoxels/MRVDBFloatGrid.h>

#pragma warning(push)
#pragma warning(disable: 4464) //relative include path contains '..' in <tbb/parallel_for.h>
#include <openvdb/tools/Filter.h>
#pragma warning(pop)

namespace MR
{


VdbVolume voxelFilter( const VdbVolume& volume, VoxelFilterType type, int width )
{
    // Copy the volume
    FloatGrid grid = std::make_shared<OpenVdbFloatGrid>( *volume.data );

    // Unfortunately, this operation does not support progress-callbacks
    openvdb::tools::Filter<openvdb::FloatGrid> filter( *grid );
    assert( ( width - 1 ) % 2 == 0 );
    const int w = ( width - 1 ) / 2;
    switch ( type )
    {
        case VoxelFilterType::Median:
            filter.median( w );
            break;
        case VoxelFilterType::Mean:
            filter.mean( w );
            break;
        case VoxelFilterType::Gaussian:
            filter.gaussian( w );
            break;
        default:
            assert( false );
    }

    // For some reason, median filter in VDB lefts some values out of range
    openvdb::tools::foreach( grid->beginValueAll(), [mn = volume.min, mx = volume.max] ( const openvdb::FloatGrid::ValueAllIter& it )
    {
        it.setValue( std::clamp( it.getValue(), mn, mx ) );
    } );

    auto mnmx = openvdb::tools::minMax( volume.data->tree() );
    VdbVolume res = volume;
    res.data = std::move( grid );
    res.min = mnmx.min();
    res.max = mnmx.max();

    return res;
}

}