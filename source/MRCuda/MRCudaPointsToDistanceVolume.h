#include "config.h"
#ifndef MRCUDA_NO_VOXELS
#include "exports.h"
#include "MRVoxels/MRVoxelsVolume.h"
#include "MRVoxels/MRPointsToDistanceVolume.h"
#include "MRVoxels/MRPointsToMeshFusion.h"

namespace MR
{

namespace Cuda
{
/// makes SimpleVolume filled with signed distances to points with normals
MRCUDA_API Expected<MR::SimpleVolumeMinMax> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params );

/// makes SimpleVolume filled with signed distances to points with normals
/// populate the volume by parts to the given callback
MRCUDA_API Expected<void> pointsToDistanceVolumeByParts( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params,
    std::function<Expected<void> ( const SimpleVolumeMinMax& volume, int zOffset )> addPart, int layerOverlap );

/// returns pointsToDistanceVolume as an object assignable to PointsToMeshParameters::createVolumeCallback;
/// the languages without implicit conversion from a callable to std::function (e.g. Python) need it, as in
///     params.createVolumeCallback = mrcudapy.pointsToDistanceVolumeCallback()
[[nodiscard]] MRCUDA_API MR::PointsToMeshParameters::CreateVolumeCallback pointsToDistanceVolumeCallback();

/// returns pointsToDistanceVolumeByParts as an object assignable to PointsToMeshParameters::createVolumeCallbackByParts;
/// the languages without implicit conversion from a callable to std::function (e.g. Python) need it, as in
///     params.createVolumeCallbackByParts = mrcudapy.pointsToDistanceVolumeByPartsCallback()
[[nodiscard]] MRCUDA_API MR::PointsToMeshParameters::CreateVolumeCallbackByParts pointsToDistanceVolumeByPartsCallback();

/// makes the subsequent pointsToMeshFusion( cloud, params ) compute the distance volume on GPU, as in
///     mrcudapy.setupPointsToMeshFusion( params )
/// the by-parts callback takes precedence, and it streams the volume instead of allocating it whole
MRCUDA_API void setupPointsToMeshFusion( MR::PointsToMeshParameters& params );

}
}
#endif