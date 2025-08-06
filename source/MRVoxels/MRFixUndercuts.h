#pragma once
#include "MRVoxelsFwd.h"

#include "MRMesh/MRVector3.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRExpected.h"
#include <functional>

namespace MR
{
namespace FixUndercuts
{
// Changes mesh:
// Fills all holes first, then:
// fixes undercuts via prolonging widest points down
// Requires to update RenderObject after using
// upDirection is in mesh space
// voxelSize -  size of voxel in mesh rasterization, precision grows with lower voxelSize
// bottomExtension - this parameter specifies how long should bottom prolongation be, if (bottomExtension <= 0) bottomExtension = 2*voxelSize
//   if mesh is not closed this is used to prolong hole and make bottom
//
// if voxelSize == 0.0f it will be counted automaticly
[[deprecated( "Use fix( mesh, params )" )]]
MRVOXELS_API MR_BIND_IGNORE Expected<void> fixUndercuts( Mesh& mesh, const Vector3f& upDirection, float voxelSize = 0.0f, float bottomExtension = 0.0f );

// Changes mesh:
// Fills all holes first, then:
// fixes undercuts (in selected area) via prolonging widest points down
// Requires to update RenderObject after using
// upDirection is in mesh space
// voxelSize -  size of voxel in mesh rasterization, precision grows with lower voxelSize
// bottomExtension - this parameter specifies how long should bottom prolongation be, if (bottomExtension <= 0) bottomExtension = 2*voxelSize
//   if mesh is not closed this is used to prolong hole and make bottom
//
// if voxelSize == 0.0f it will be counted automaticly
[[deprecated( "Use fix( mesh, params )" )]]
MRVOXELS_API MR_BIND_IGNORE Expected<void> fixUndercuts( Mesh& mesh, const FaceBitSet& selectedArea, const Vector3f& upDirection, float voxelSize = 0.0f, float bottomExtension = 0.0f );

/// Parameters that is used to find undercuts
struct FindParams
{
    /// Primitives that are not visible from up direction are considered as undercuts (fix undercuts is performed downwards (in `-direction`))
    Vector3f upDirection;

    /// vertical angle of fixed undercut walls (note that this value is approximate - it defines "camera" position for internal projective transformation)
    /// 0 - strictly vertical walls of undercuts area
    /// positive - expanding downwards walls
    /// negative - shrinking downwards walls
    float wallAngle = 0.0f;
};

/// Fix undercuts function parameters
struct FixParams
{
    /// parameters of what is considered as undercut
    FindParams findParameters;

    /// voxel size for internal computations: lower size - better precision but more system resources required
    float voxelSize = 0.0f;

    /// minimum extension of bottom part of the mesh
    float bottomExtension = 0.0f;

    /// if set - only this region will be fixed (but still all mesh will be rebuild)
    const FaceBitSet* region = nullptr;

    /// if true applies one iterations of gaussian filtering for voxels, useful if thin walls expected
    bool smooth = false;

    ProgressCallback cb;
};

/// Fixes undercut areas by building vertical walls under it,
/// algorithm is performed in voxel space, so the mesh is completely rebuilt after this operation
MRVOXELS_API Expected<void> fix( Mesh& mesh, const FixParams& params );

// Input - undercut faces, insertion direction
// Output - metric value
using UndercutMetric = std::function<double( const FaceBitSet&, const FindParams& params )>;

/// returns the metric that computes total area of undercut faces
[[nodiscard]] MRVOXELS_API UndercutMetric getUndercutAreaMetric( const Mesh& mesh );

/// returns the metric that computes summed absolute projected area of undercut
[[nodiscard]] MRVOXELS_API UndercutMetric getUndercutAreaProjectionMetric( const Mesh& mesh );

/// Adds to \param outUndercuts undercut faces
[[deprecated( "Use find( mesh, params)" )]]
MRVOXELS_API MR_BIND_IGNORE void findUndercuts( const Mesh& mesh, const Vector3f& upDirection, FaceBitSet& outUndercuts );
/// Adds to \param outUndercuts undercut vertices
[[deprecated( "Use find( mesh, params )" )]]
MRVOXELS_API MR_BIND_IGNORE void findUndercuts( const Mesh& mesh, const Vector3f& upDirection, VertBitSet& outUndercuts );

/// Adds to \param outUndercuts undercut faces
/// Returns summary metric of undercut faces
[[deprecated( "Use find( mesh, params, metric )" )]]
[[nodiscard]] MRVOXELS_API MR_BIND_IGNORE double findUndercuts( const Mesh& mesh, const Vector3f& upDirection, FaceBitSet& outUndercuts, const UndercutMetric& metric );

/// Adds undercuts to \param outUndercuts
/// if metric is set returns metric of found undercuts, otherwise returns DBL_MAX
MRVOXELS_API double find( const Mesh& mesh, const FindParams& params, FaceBitSet& outUndercuts, const UndercutMetric& metric = {} );
/// Adds undercuts to \param outUndercuts
MRVOXELS_API void find( const Mesh& mesh, const FindParams& params, VertBitSet& outUndercuts );

/// Fast score undercuts projected area via distance map with given resolution
/// lower resolution means lower precision, but faster work
/// \note does not support wallAngle yet
[[nodiscard]] MRVOXELS_API double scoreUndercuts( const Mesh& mesh, const Vector3f& upDirection, const Vector2i& resolution );

struct ImproveDirectionParameters
{
    // Hint direction which will be improved
    Vector3f hintDirection;
    // Radial step given in radians look improveDirection comment
    float baseAngleStep{5.0f*PI_F / 180.0f};
    // Maximum radial line given in radians look improveDirection comment
    float maxBaseAngle{30.0f*PI_F / 180.0f};
    // Polar angle step
    float polarAngleStep{20.0f*PI_F / 180.0f};
};

struct DistMapImproveDirectionParameters : ImproveDirectionParameters
{
    // Resolution of distance map, lower it is, faster score works
    Vector2i distanceMapResolution{100,100};
};

// Parallel finds best of several directions defined by ImproveDirectionParameters struct
/// \note does not support wallAngle yet
///                      ________
///        Top view:    /  \__/  \-----> maximum radial line   Side view:  |    /    _/
///                    /  / \/ \  \                                        |   /   _/ - maxBaseAngle
///                   |--|------|--|                                       |  /  _/     difference between two angles is baseAngleStep
///                    \  \_/\_/  /                                        | / _/
///                     \__/__\__/                                         |/_/
/// This picture shows polarAngle = 60 deg
[[nodiscard]] MRVOXELS_API Vector3f improveDirection( const Mesh& mesh, const ImproveDirectionParameters& params, const UndercutMetric& metric );
/// Score candidates with distance maps, lower resolution -> faster score
/// \note does not support wallAngle yet
[[nodiscard]] MRVOXELS_API Vector3f distMapImproveDirection( const Mesh& mesh, const DistMapImproveDirectionParameters& params );
}
}
