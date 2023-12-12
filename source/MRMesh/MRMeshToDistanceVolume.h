#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRProgressCallback.h"
#include "MRSignDetectionMode.h"
#include "MRSimpleVolume.h"
#include "MRExpected.h"
#include <cfloat>
#include <memory>

namespace MR
{

struct DistanceVolumeParams
{
    /// origin point of voxels box
    Vector3f origin;
    /// progress callback
    ProgressCallback cb;
    /// size of voxel on each axis
    Vector3f voxelSize{ 1.0f,1.0f,1.0f };
    /// num voxels along each axis
    Vector3i dimensions{ 100,100,100 };
};

struct MeshToDistanceVolumeParams : DistanceVolumeParams
{
    /// minimum squared value in a voxel
    float minDistSq{ 0 };
    /// maximum squared value in a voxel
    float maxDistSq{ FLT_MAX };
    /// the method to compute distance sign
    SignDetectionMode signMode{ SignDetectionMode::ProjectionNormal };

    std::shared_ptr<IFastWindingNumber> fwn;
};
using MeshToSimpleVolumeParams [[deprecated]] = MeshToDistanceVolumeParams;

/// makes SimpleVolume filled with (signed or unsigned) distances from Mesh with given settings
MRMESH_API Expected<SimpleVolume, std::string> meshToDistanceVolume( const Mesh& mesh, const MeshToDistanceVolumeParams& params = {} );

/// returns a volume filled with the values:
/// v < 0: this point is within offset distance to region-part of mesh and it is closer to region-part than to not-region-part
MRMESH_API Expected<SimpleVolume, std::string> meshRegionToIndicatorVolume( const Mesh& mesh, const FaceBitSet& region,
    float offset, const DistanceVolumeParams& params );

}
