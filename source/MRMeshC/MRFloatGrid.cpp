#include "MRFloatGrid.h"
#include "MRBitSet.h"

#include "detail/TypeCast.h"

#include "MRVoxels/MRFloatGrid.h"

using namespace MR;

REGISTER_AUTO_CAST( FloatGrid )
REGISTER_AUTO_CAST( VoxelBitSet )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( Vector3i )
REGISTER_AUTO_CAST( Box3i )

MRFloatGrid* mrFloatGridResampledUniformly( const MRFloatGrid* grid_, float voxelScale, MRProgressCallback cb )
{
    ARG( grid );
    RETURN_NEW( resampled( grid, voxelScale, cb ) );
}

MRFloatGrid* mrFloatGridResampled( const MRFloatGrid* grid_, const MRVector3f* voxelScale_, MRProgressCallback cb )
{
    ARG( grid ); ARG( voxelScale );
    RETURN_NEW( resampled( grid, voxelScale, cb ) );
}

MRFloatGrid* mrFloatGridCropped( const MRFloatGrid* grid_, const MRBox3i* box_, MRProgressCallback cb )
{
    ARG(grid); ARG(box);
    RETURN_NEW( cropped( grid, box, cb ) );
}

float mrFloatGridGetValue( const MRFloatGrid* grid_, const MRVector3i* p_ )
{
    ARG( grid ); ARG( p );
    return getValue( grid, p );
}

void mrFloatGridSetValue( MRFloatGrid* grid_, const MRVector3i* p_, float value )
{
    ARG( grid ); ARG( p );
    setValue( grid, p, value );
}

void mrFloatGridSetValueForRegion( MRFloatGrid* grid_, const MRVoxelBitSet* region_, float value )
{
    ARG( grid ); ARG( region );
    setValue( grid, region, value );
}