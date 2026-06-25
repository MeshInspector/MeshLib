#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRPolyline.h"

#include <vector>

namespace MR
{

struct EndMillTool;

/// Compute bounding box for swept volume for given tool and toolpath
MRVOXELS_API Box3f computeWorkArea( const Polyline3& toolpath, const MeshPart& tool );

/// Compute required voxel volume's dimensions for given work area
MRVOXELS_API Box3i computeGridBox( const Box3f& workArea, float voxelSize );

/// Parameters for computeSweptVolume* functions
struct ComputeSweptVolumeParameters
{
    /// toolpath
    const Polyline3& path;
    /// tool mesh
    MeshPart toolMesh;
    /// tool specifications, can be used for more precise computations
    /// the tool spec and the tool mesh are expected to relate to the same tool
    /// if omitted, tool mesh is used
    const EndMillTool* toolSpec = nullptr;
    /// voxel size for internal voxel volumes
    // TODO: replace with tolerance and make the voxel size implementation-specific
    float voxelSize{ 0.0f };
    /// (distance volume) max memory amount used for the distance volume, zero for no limits
    size_t memoryLimit = 0;
    /// progress callback
    ProgressCallback cb;
};

/// Compute swept volume for given toolpath and tool
/// Builds mesh for each tool movement and joins them using voxel boolean
MRVOXELS_API Expected<Mesh> computeSweptVolumeWithMeshMovement( const ComputeSweptVolumeParameters& params );

/// Compute swept volume for given toolpath and tool
/// Creates a distance-to-tool volume and converts it to mesh using the marching cubes algorithm
MRVOXELS_API Expected<Mesh> computeSweptVolumeWithDistanceVolume( const ComputeSweptVolumeParameters& params );

/// Interface for custom tool distance computation implementations
class MRVOXELS_CLASS IComputeToolDistance
{
public:
    virtual ~IComputeToolDistance() = default;

    /// Prepare for a voxel grid of given dims and copy tool path and tool spec data
    /// \return Maximum dimensions that can be processed at once (e.g. due to memory limits)
    virtual Expected<Vector3i> prepare( const Vector3i& dims, const Polyline3& toolpath, const EndMillTool& toolSpec ) = 0;
    /// Prepare for a voxel grid of given dims and copy tool path and tool spec data
    /// \return Maximum dimensions that can be processed at once (e.g. due to memory limits)
    virtual Expected<Vector3i> prepare( const Vector3i& dims, const Polyline3& toolpath, const Polyline2& toolProfile ) = 0;

    /// Compute tool distance
    virtual Expected<void> computeToolDistance(
        std::vector<float>& output,
        const Vector3i& dims, float voxelSize, const Vector3f& origin,
        float padding
    ) const = 0;
    // TODO: async
};

/// Compute swept volume for given toolpath and tool
/// Creates a distance-to-tool volume using custom tool distance computation object and converts it to mesh using
/// the marching cubes algorithm
MRVOXELS_API Expected<Mesh> computeSweptVolumeWithCustomToolDistance( IComputeToolDistance& comp, const ComputeSweptVolumeParameters& params );

} // namespace MR
