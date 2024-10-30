#pragma once

#include "MRDistanceVolumeParams.h"
#include "MRVoxelsVolume.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRMeshDistance.h"
#include <memory>

namespace MR
{

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
