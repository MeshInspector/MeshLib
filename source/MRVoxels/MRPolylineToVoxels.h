#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRVoxelsFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRVDBConversions.h"
#include "MRDistanceVolumeParams.h"
#include "MRMesh/MRDistanceToMeshOptions.h"
#include "MRMesh/MRAffineXf3.h"

namespace MR
{

struct PolylineToDistanceVolumeParams
{
    Vector3f voxelSize = Vector3f::diagonal( 1.f );
    /// offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
    float offsetCount = 3;
    AffineXf3f worldXf; // line initial transform
    AffineXf3f* outXf{ nullptr }; // optional output: xf to original mesh (respecting worldXf)
    ProgressCallback cb;
};

/// convert polyline to voxels distance field
MRVOXELS_API Expected<FloatGrid> polylineToDistanceField( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params );

/// convert polyline to VDB volume
MRVOXELS_API Expected<VdbVolume> polylineToVdbVolume( const Polyline3& polyline, const PolylineToDistanceVolumeParams& params );

/// Settings to conversion polyline to volume
struct PolylineToVolumeParams
{
    DistanceVolumeParams vol;

    DistanceToMeshOptions dist;
};

/// convert polyline to simple volume
MRVOXELS_API Expected<SimpleVolume> polylineToSimpleVolume( const Polyline3& polyline, const PolylineToVolumeParams& params );

/// convert polyline to function volume
MRVOXELS_API Expected<FunctionVolume> polylineToFunctionVolume( const Polyline3& polyline, const PolylineToVolumeParams& params );

}
