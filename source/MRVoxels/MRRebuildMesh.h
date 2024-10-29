#pragma once

#include "MRVoxelsFwd.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MREnums.h"
#include "MRMesh/MRSignDetectionMode.h"

namespace MR
{

struct RebuildMeshSettings
{
    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    SignDetectionModeShort signMode = SignDetectionModeShort::Auto;

    OffsetMode offsetMode = OffsetMode::Standard;

    /// if non-null then created sharp edges (only if offsetMode = OffsetMode::Sharpening) will be saved here
    UndirectedEdgeBitSet* outSharpEdges = nullptr;

    /// if general winding number is used to differentiate inside from outside:
    /// positive distance if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.5f;

    /// if general winding number is used to differentiate inside from outside:
    /// determines the precision of fast approximation: the more the better, minimum value is 1
    float windingNumberBeta = 2;

    /// defines particular implementation of IFastWindingNumber interface that will compute windings (if required).
    /// If it is not specified, default FastWindingNumber is used
    std::shared_ptr<IFastWindingNumber> fwn;

    /// whether to decimate resulting mesh
    bool decimate = true;

    /// only if decimate = true:
    /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
    float tinyEdgeLength = -1;

    /// To report algorithm's progress and cancel it on user demand
    ProgressCallback progress;

    /// this callback is invoked when SignDetectionMode is determined (useful if signMode = SignDetectionModeShort::Auto),
    /// but before actual work begins
    std::function<void(SignDetectionMode)> onSignDetectionModeSelected;
};

/// fixes all types of issues in input mesh (degenerations, holes, self-intersections, etc.)
/// by first converting mesh in voxel representation, and then backward
[[nodiscard]] MRVOXELS_API Expected<Mesh> rebuildMesh( const MeshPart& mp, const RebuildMeshSettings& settings );

} //namespace MR
