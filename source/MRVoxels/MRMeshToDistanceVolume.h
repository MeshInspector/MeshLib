#pragma once

#include "MRDistanceVolumeParams.h"
#include "MRMesh/MRSignDetectionMode.h"
#include "MRVoxelsVolume.h"
#include "MRMesh/MRExpected.h"
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

    /// only for SignDetectionMode::HoleWindingRule:
    /// positive distance if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.5f;

    /// only for SignDetectionMode::HoleWindingRule:
    /// determines the precision of fast approximation: the more the better, minimum value is 1
    float windingNumberBeta = 2;
};

/// computes signed distance from point (p) to mesh part (mp) following options (op)
[[nodiscard]] MRVOXELS_API std::optional<float> signedDistanceToMesh( const MeshPart& mp, const Vector3f& p, const DistanceToMeshOptions& op );

struct MeshToDistanceVolumeParams
{
    DistanceVolumeParams vol;

    DistanceToMeshOptions dist;

    std::shared_ptr<IFastWindingNumber> fwn;
};

/// makes SimpleVolume filled with (signed or unsigned) distances from Mesh with given settings
MRVOXELS_API Expected<SimpleVolumeMinMax> meshToDistanceVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params = {} );

/// makes FunctionVolume representing (signed or unsigned) distances from Mesh with given settings
MRVOXELS_API FunctionVolume meshToDistanceFunctionVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params = {} );

/// returns a volume filled with the values:
/// v < 0: this point is within offset distance to region-part of mesh and it is closer to region-part than to not-region-part
MRVOXELS_API Expected<SimpleVolumeMinMax> meshRegionToIndicatorVolume( const Mesh& mesh, const FaceBitSet& region,
    float offset, const DistanceVolumeParams& params );


struct MeshToDirectionVolumeParams
{
    DistanceVolumeParams vol;
    DistanceToMeshOptions dist; // note that signMode is ignored in this algorithm
    std::shared_ptr<IPointsToMeshProjector> projector;
};

/// Converts mesh into 4d voxels, so that each cell in 3d space holds the direction from the closest point on mesh to the cell position.
/// Resulting volume is encoded by 3 separate 3d volumes, corresponding to `x`, `y` and `z` components of vectors respectively.
/// \param params Expected to have valid (not null) projector, with invoked method \ref IPointsToMeshProjector::updateMeshData
MRVOXELS_API Expected<std::array<SimpleVolumeMinMax, 3>> meshToDirectionVolume( const MeshToDirectionVolumeParams& params );

} //namespace MR
