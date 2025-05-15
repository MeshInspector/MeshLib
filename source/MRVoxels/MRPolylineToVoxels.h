#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRVoxelsFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRVDBConversions.h"

namespace MR
{

/// convert polyline to voxels distance field
/// \param offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
MRVOXELS_API Expected<FloatGrid> polylineToDistanceField( const Polyline3& polyline, const Vector3f& voxelSize, float offsetCount = 3, ProgressCallback cb = {} );

/// convert polyline to VDB volume
/// \param offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
MRVOXELS_API Expected<VdbVolume> polylineToVdbVolume( const Polyline3& polyline, const Vector3f& voxelSize, float offsetCount = 3, ProgressCallback cb = {} );

/// convert polyline to simple volume
/// \param offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
MRVOXELS_API Expected<SimpleVolume> polylineToSimpleVolume( const Polyline3& polyline, const Vector3f& voxelSize, float offsetCount = 3, ProgressCallback cb = {} );


/// Settings to conversion polyline to function volume
struct PolylineToDistanceVolumeParams
{
    DistanceVolumeParams vol;

    DistanceToMeshOptions dist;
};

/// convert polyline to function volume
MRVOXELS_API Expected<FunctionVolume> polylineToFunctionVolume( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params );

}
