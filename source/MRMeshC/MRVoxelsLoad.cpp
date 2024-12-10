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

MRVdbVolumes* mrVoxelsLoadFromAnySupportedFormat( const char* file, MRProgressCallback cb_, MRString** errorStr )
{
    auto res = VoxelsLoad::fromAnySupportedFormat( file, cb_ );

    if ( res )
    {
        std::vector<MRVdbVolume> volumes( res->size() );
        
        for ( size_t i = 0; i < res->size(); ++i )
        {
            volumes[i].data = auto_cast( new_from( std::move( ( *res )[i].data ) ) );
            volumes[i].dims = auto_cast( ( *res )[i].dims );
            volumes[i].voxelSize = auto_cast( ( *res )[i].voxelSize );
            volumes[i].min = ( *res )[i].min;
            volumes[i].max = ( *res )[i].max;
        }
        
        return (MRVdbVolumes*)( NEW_VECTOR( std::move( volumes ) ) );
    }

    if ( errorStr && !res )
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;
}