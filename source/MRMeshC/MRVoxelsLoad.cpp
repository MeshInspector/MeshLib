#include "MRVoxelsLoad.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRVoxels/MRVoxelsLoad.h"

using namespace MR;
REGISTER_AUTO_CAST2( std::string, MRString )
REGISTER_AUTO_CAST( FloatGrid )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( Vector3i )
REGISTER_AUTO_CAST( VdbVolumes )

const MRVdbVolume mrVdbVolumesGet( const MRVdbVolumes* volumes_, size_t index )
{
    ARG( volumes );
    const auto& result = volumes[index];
    MRVdbVolume res;
    res.data = ( MRFloatGrid* )&result.data;
    res.dims = auto_cast( result.dims );
    res.voxelSize = auto_cast( result.voxelSize );
    res.min = result.min;
    res.max = result.max;
    return res;    
}

size_t mrVdbVolumesSize( const MRVdbVolumes* volumes_ )
{
    ARG( volumes );
    return volumes.size();
}

void mrVdbVolumesFree( MRVdbVolumes* volumes_ )
{
    ARG_PTR( volumes );
    delete volumes;
}

MRVdbVolumes* mrVoxelsLoadFromAnySupportedFormat( const char* file, MRProgressCallback cb, MRString** errorStr )
{
    if ( auto res = VoxelsLoad::fromAnySupportedFormat( file, cb ) )    
        RETURN_NEW( std::move( res.value() ) );    
    else if ( errorStr )
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;
}