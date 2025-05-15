#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRVoxelsFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRVDBConversions.h"
#include "MRDistanceVolumeParams.h"
#include "MRMesh/MRDistanceToMeshOptions.h"

namespace MR
{

struct PolylineToDistanceVolumeParams
{
    const Vector3f& voxelSize = Vector3f::diagonal( 1.f );
    /// offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
    float offsetCount = 3;
    ProgressCallback cb = {};
};

/// convert polyline to voxels distance field
MRVOXELS_API Expected<FloatGrid> polylineToDistanceField( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params );

/// convert polyline to VDB volume
/// \param offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
MRVOXELS_API Expected<VdbVolume> polylineToVdbVolume( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params );

/// convert polyline to simple volume
/// \param offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
MRVOXELS_API Expected<SimpleVolume> polylineToSimpleVolume( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params );


/// Settings to conversion polyline to function volume
struct PolylineToFunctionVolumeParams
{
    DistanceVolumeParams vol;

    DistanceToMeshOptions dist;
};

/// convert polyline to function volume
MRVOXELS_API Expected<FunctionVolume> polylineToFunctionVolume( const Polyline3& polyline, const PolylineToFunctionVolumeParams& params );

}
