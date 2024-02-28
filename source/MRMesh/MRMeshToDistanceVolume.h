#pragma once

#include "MRDistanceVolumeParams.h"
#include "MRSignDetectionMode.h"
#include "MRSimpleVolume.h"
#include "MRExpected.h"
#include <cfloat>
#include <memory>

namespace MR
{

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

/// makes SimpleVolume filled with (signed or unsigned) distances from Mesh with given settings
MRMESH_API Expected<SimpleVolume, std::string> meshToDistanceVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params = {} );

/// makes FunctionVolume representing (signed or unsigned) distances from Mesh with given settings
MRMESH_API Expected<FunctionVolume> meshToDistanceFunctionVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params = {} );

/// returns a volume filled with the values:
/// v < 0: this point is within offset distance to region-part of mesh and it is closer to region-part than to not-region-part
MRMESH_API Expected<SimpleVolume, std::string> meshRegionToIndicatorVolume( const Mesh& mesh, const FaceBitSet& region,
    float offset, const DistanceVolumeParams& params );

} //namespace MR
