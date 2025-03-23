#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

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

struct VariadicOffsetParams
{
    /// stop searching as soon as any point within this distance is found
    float minDistance = 0;

    /// find the closest point only if the distance to it is less than given value
    float maxDistance = 0;

    /// maximal weight among all points in the cloud;
    /// if this value is imprecise, then more computations will be made by algorithm
    float maxWeight = 0;

    /// maximal magnitude of gradient of points' weight in the cloud, >=0;
    /// if maxWeightGrad < 1 then more search optimizations can be done
    float maxWeightGrad = 0;
};

/// consider a point cloud where each point has additive weight,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// finds the point with minimal distance to given 3D location
[[nodiscard]] MRMESH_API PointAndDistance findClosestWeightedPoint( const Vector3f & loc,
    const AABBTreePoints& tree, const VertMetric& pointWeights, const VariadicOffsetParams& params );

} //namespace MR
