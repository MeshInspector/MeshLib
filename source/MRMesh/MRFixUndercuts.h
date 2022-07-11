#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRVector2.h"
#include "MRConstants.h"
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
MRMESH_API void fixUndercuts( Mesh& mesh, const Vector3f& upDirection, float voxelSize = 0.0f, float bottomExtension = 0.0f );

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
MRMESH_API void fixUndercuts( Mesh& mesh, const FaceBitSet& selectedArea, const Vector3f& upDirection, float voxelSize = 0.0f, float bottomExtension = 0.0f );

// Input - undercut faces, insertion direction
// Output - metric value
using UndercutMetric = std::function<double( const FaceBitSet&, const Vector3f& upDir )>;

// returns simple area of undercut faces
MRMESH_API UndercutMetric getUndercutAreaMetric( const Mesh& mesh );
// returns projected are of undercut faces
MRMESH_API UndercutMetric getUndercutAreaProjectionMetric( const Mesh& mesh );

// Adds to outUndercuts undercut faces
// Returns summary metric of undercut faces
MRMESH_API double findUndercuts( const Mesh& mesh, const Vector3f& upDirection, FaceBitSet& outUndercuts, const UndercutMetric& metric );
// the same but search for vertices
MRMESH_API void findUndercuts( const Mesh& mesh, const Vector3f& upDirection, VertBitSet& outUndercuts );
// Fast score undercuts projected area via distance map with given resolution
// lower resolution means lower precision, but faster work
MRMESH_API double scoreUndercuts( const Mesh& mesh, const Vector3f& upDirection, const Vector2i& resolution );

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
//                      ________ 
//        Top view:    /  \__/  \-----> maximum radial line   Side view:  |    /    _/
//                    /  / \/ \  \                                        |   /   _/ - maxBaseAngle
//                   |--|------|--|                                       |  /  _/     difference between two angles is baseAngleStep
//                    \  \_/\_/  /                                        | / _/
//                     \__/__\__/                                         |/_/
// This picture shows polarAngle = 60 deg
MRMESH_API Vector3f improveDirection( const Mesh& mesh, const ImproveDirectionParameters& params, const UndercutMetric& metric );
// Score candidates with distance maps, lower resolution -> faster score
MRMESH_API Vector3f distMapImproveDirection( const Mesh& mesh, const DistMapImproveDirectionParameters& params );
}
}
#endif