#pragma once
#include "MRMeshFwd.h"
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRVector3.h"
#include "MRProgressCallback.h"
#include "MRSignDetectionMode.h"
#include "MRExpected.h"
#include <cfloat>
#include <memory>

namespace MR
{

struct MeshToDistanceVolumeParams
{
    /// origin point of voxels box
    Vector3f origin;
    /// progress callback
    ProgressCallback cb;
    /// size of voxel on each axis
    Vector3f voxelSize{ 1.0f,1.0f,1.0f };
    /// num voxels along each axis
    Vector3i dimensions{ 100,100,100 };
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
[[deprecated( "use meshToDistanceVolume()" )]] MRMESH_API Expected<SimpleVolume, std::string> meshToSimpleVolume( const Mesh& mesh, const MeshToDistanceVolumeParams& params = {} );

}
#endif
