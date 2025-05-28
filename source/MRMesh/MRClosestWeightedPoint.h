#pragma once

#include "MRMeshFwd.h"
#include "MRMeshTriPoint.h"
#include "MRPch/MRBindingMacros.h"
#include <cfloat>

namespace MR
{

struct PointAndDistance
{
    /// a cloud's point
    VertId vId;

    /// the distance from input location to point vId considering point's weight
    float dist = 0;

    /// check for validity, otherwise there is no point closer than maxBidirDist
    [[nodiscard]] bool valid() const { return vId.valid(); }
    [[nodiscard]] explicit operator bool() const { return vId.valid(); }
};

struct MeshPointAndDistance
{
    /// point location
    Vector3f loc;

    /// the corresponding point on mesh in barycentric representation
    MeshTriPoint mtp;

    /// euclidean distance from input location to mtp
    float eucledeanDist = 0;

    /// point's weight
    float w = 0;

    /// either
    /// 1) bidirectional distances are computed, or
    /// 2) input location is locally outside of the surface (by pseudonormal)
    /// used for optimization
    bool bidirectionalOrOutside = true;

    /// bidirectional distance from input location to mtp considering point's weight
    [[nodiscard]] float bidirDist() const
    {
        return eucledeanDist - w;
    }

    /// the distance from input location to mtp considering point's weight and location inside/outside;
    /// dist() is continuous function of location unlike innerDist(),
    /// which makes 2*weight jump if the location moves through the surface
    [[nodiscard]] float dist() const
    {
        return ( bidirectionalOrOutside ? eucledeanDist : -eucledeanDist ) - w;
    }

    /// comparison telling which point is closer to the location
    auto operator <=> ( const MeshPointAndDistance& other ) const
    {
        if ( bidirectionalOrOutside && other.bidirectionalOrOutside  )
            return eucledeanDist - w <=> other.eucledeanDist - other.w;
        if ( bidirectionalOrOutside != other.bidirectionalOrOutside )
            return eucledeanDist <=> other.eucledeanDist;
        return eucledeanDist + ( bidirectionalOrOutside ? -w : w ) <=>
                other.eucledeanDist + ( other.bidirectionalOrOutside ? -other.w : other.w );
    }

    /// check for validity, otherwise there is no point closer than maxBidirDist
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

    /// for points, it must always true;
    /// for triangles:
    ///   if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
    ///   if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
    bool bidirectionalMode = true;

    // To allow passing Python lambdas into `pointWeight`.
    MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
};

struct DistanceFromWeightedPointsComputeParams : DistanceFromWeightedPointsParams
{
    /// stop searching as soon as any point within this weighted distance is found,
    /// ignored if bidirectionalMode=false, because in one-directional mode, weighted distance approximation not only decreases
    float minBidirDist = -FLT_MAX; // default 0 here does not work for negative distances

    /// find the closest point only if weighted bidirectional distance to it is less than given value
    float maxBidirDist = FLT_MAX;
};

/// consider a point cloud where each point has additive weight,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// finds the point with minimal distance to given 3D location
[[nodiscard]] MRMESH_API PointAndDistance findClosestWeightedPoint( const Vector3f& loc,
    const AABBTreePoints& tree, const DistanceFromWeightedPointsComputeParams& params );

/// consider a mesh where each vertex has additive weight, and this weight is linearly interpolated in mesh triangles,
/// and the distance to a point is considered equal to (euclidean distance - weight),
/// finds the point on given mesh with minimal distance to given 3D location
[[nodiscard]] MRMESH_API MeshPointAndDistance findClosestWeightedMeshPoint( const Vector3f& loc,
    const Mesh& mesh, const DistanceFromWeightedPointsComputeParams& params );

} //namespace MR
