#pragma once

#include "MRDistanceVolumeParams.h"
#include "MRSignDetectionMode.h"
#include "MRVoxelsVolume.h"
#include "MRExpected.h"
#include <cfloat>
#include <memory>
#include <optional>

namespace MR
{

struct DistanceToMeshOptions
{
    /// minimum squared distance from a point to mesh
    float minDistSq{ 0 };

    /// maximum squared distance from a point to mesh
    float maxDistSq{ FLT_MAX };

    /// the method to compute distance sign
    SignDetectionMode signMode{ SignDetectionMode::ProjectionNormal };
};

/// computes signed distance from point (p) to mesh part (mp) following options (op)
[[nodiscard]] MRMESH_API std::optional<float> signedDistanceToMesh( const MeshPart& mp, const Vector3f& p, const DistanceToMeshOptions& op );

struct MeshToDistanceVolumeParams
{
    DistanceVolumeParams vol;

    DistanceToMeshOptions dist;

    std::shared_ptr<IFastWindingNumber> fwn;
};

/// makes SimpleVolume filled with (signed or unsigned) distances from Mesh with given settings
MRMESH_API Expected<SimpleVolume> meshToDistanceVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params = {} );

/// makes FunctionVolume representing (signed or unsigned) distances from Mesh with given settings
MRMESH_API FunctionVolume meshToDistanceFunctionVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params = {} );

/// converts function volume into simple volume
MRMESH_API Expected<SimpleVolume> functionVolumeToSimpleVolume( const FunctionVolume& volume, const ProgressCallback& callback = {} );

/// returns a volume filled with the values:
/// v < 0: this point is within offset distance to region-part of mesh and it is closer to region-part than to not-region-part
MRMESH_API Expected<SimpleVolume> meshRegionToIndicatorVolume( const Mesh& mesh, const FaceBitSet& region,
    float offset, const DistanceVolumeParams& params );

} //namespace MR
