#include "MRVoxelsSave.h"

#include "detail/TypeCast.h"

#include "MRVoxels/MRVoxelsFwd.h"
#include "MRVoxels/MRVoxelsSave.h"

using namespace MR;
REGISTER_AUTO_CAST2( std::string, MRString )
REGISTER_AUTO_CAST( FloatGrid )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( Vector3i )

void mrVoxelsSaveToAnySupportedFormat( const MRVdbVolume* volume_, const char* file, MRProgressCallback cb_, MRString** errorStr )
{
    auto cb = [cb_] ( float progress ) -> bool { return cb_( progress ); };
    VdbVolume volume;
    volume.data = auto_cast( *volume_->data );
    volume.dims = auto_cast(volume_->dims);
    volume.voxelSize = auto_cast( volume_->voxelSize );
    volume.min = volume_->min;
    volume.max = volume_->max;

    auto res = cb_ ? VoxelsSave::toAnySupportedFormat(volume, file, cb) : VoxelsSave::toAnySupportedFormat(volume, file);
    if ( !res && errorStr )
    {
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );
    }
}