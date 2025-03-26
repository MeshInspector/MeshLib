#pragma once

#include "MRMeshFwd.h"
#include "MRMeshTriPoint.h"
#include <cfloat>

namespace MR
{

struct PointAndDistance
{
    /// a cloud's point
    VertId vId;

    /// the distance from input location to point vId considering point's weight
    float dist = 0;

    /// check for validity, otherwise there is no point closer than maxDistance
    [[nodiscard]] bool valid() const { return vId.valid(); }
    [[nodiscard]] explicit operator bool() const { return vId.valid(); }
};

struct MeshPointAndDistance
{
    /// a point on mesh in barycentric representation
    MeshTriPoint mtp;

    /// the distance from input location to mtp considering point's weight
    float dist = 0;

    /// check for validity, otherwise there is no point closer than maxDistance
    [[nodiscard]] bool valid() const { return mtp.valid(); }
    [[nodiscard]] explicit operator bool() const { return mtp.valid(); }
};

struct DistanceFromWeightedPointsParams
{
    /// function returning the weight of each point, must be set by the user
    VertMetric pointWeight;

    /// maximal weight among all points in the cloud;
    /// if this value is imprecise, then more computations will be made by algorithm
    float maxWeight = 0;

    /// maximal magnitude of gradient of points' weight in the cloud, >=0;
    /// if maxWeightGrad < 1 then more search optimizations can be done
    float maxWeightGrad = FLT_MAX;
};

struct DistanceFromWeightedPointsComputeParams : DistanceFromWeightedPointsParams
{
    /// stop searching as soon as any point within this distance is found
    float minDistance = 0;

    /// find the closest point only if the distance to it is less than given value
    float maxDistance = FLT_MAX;
};

/// consider a point cloud where each point has additive weight,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// finds the point with minimal distance to given 3D location
[[nodiscard]] MRMESH_API PointAndDistance findClosestWeightedPoint( const Vector3f& loc,
    const AABBTreePoints& tree, const DistanceFromWeightedPointsComputeParams& params );

/// consider a mesh where each vertex has additive weight, and this weight is linearly interpolated in mesh triangles,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// finds the point on given mesh part with minimal distance to given 3D location
[[nodiscard]] MRMESH_API MeshPointAndDistance findClosestWeightedMeshPoint( const Vector3f& loc,
    const Mesh& mesh, const DistanceFromWeightedPointsComputeParams& params );

} //namespace MR
