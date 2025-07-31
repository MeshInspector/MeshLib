#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRId.h"
#include "MRPointCloudPart.h"
#include <cfloat>

namespace MR
{
/// \addtogroup AABBTreeGroup
/// \{

struct PointsProjectionResult
{
    /// squared distance from pt to proj
    float distSq{ 0 };
    /// the closest vertex in point cloud
    VertId vId;

    auto operator <=>(const PointsProjectionResult &) const = default;
};


/**
 * \brief computes the closest point on point cloud to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
 * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 * \param skipCb callback to discard VertId projection candidate
 */
[[nodiscard]] MRMESH_API PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const PointCloudPart& pcp,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f* xf = nullptr,
    float loDistLimitSq = 0,
    VertPredicate skipCb = {} );

/**
 * \brief computes the closest point on AABBTreePoints to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
 * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 * \param region if not nullptr, all points not from the given region will be ignored
 * \param skipCb callback to discard VertId projection candidate
 */
[[nodiscard]] MRMESH_API PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const AABBTreePoints& tree,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f* xf = nullptr,
    float loDistLimitSq = 0,
    const VertBitSet * region = nullptr,
    VertPredicate skipCb = {} );

/**
 * \brief finds a number of the closest points in the cloud (as configured in \param res) to given point
 * \param upDistLimitSq upper limit on the distance in question, points with larger distance than it will not be returned
 * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, the algorithm can return given number of points within this distance even skipping closer ones
 */
MRMESH_API void findFewClosestPoints( const Vector3f& pt, const PointCloud& pc, FewSmallest<PointsProjectionResult> & res,
    float upDistLimitSq = FLT_MAX,
    const AffineXf3f* xf = nullptr,
    float loDistLimitSq = 0 );

/**
 * \brief finds given number of closest points (excluding itself) to each valid point in the cloud;
 * \param numNei the number of closest points to find for each point
 * \return a buffer where for every valid point with index `i` its neighbours are stored at indices [i*numNei; (i+1)*numNei)
 */
[[nodiscard]] MRMESH_API Buffer<VertId> findNClosestPointsPerPoint( const PointCloud& pc, int numNei, const ProgressCallback & progress = {} );

/// finds two closest points (first id < second id) in whole point cloud
[[nodiscard]] MRMESH_API VertPair findTwoClosestPoints( const PointCloud& pc, const ProgressCallback & progress = {} );

/// settings for \ref IPointsProjector::findProjections
struct FindProjectionOnPointsSettings
{
    /// bitset of valid input points
    const BitSet* valid = nullptr;
    /// affine transformation for input points
    const AffineXf3f* xf = nullptr;
    /// upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
    float upDistLimitSq = FLT_MAX;
    /// low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    float loDistLimitSq = 0.f;
    /// if true, discards a projection candidate with the same index as the target point
    bool skipSameIndex = false;
    /// progress callback
    ProgressCallback cb;
};

/// abstract class for computing the closest points of point clouds
class IPointsProjector
{
public:
    virtual ~IPointsProjector() = default;

    /// sets the reference point cloud
    virtual Expected<void> setPointCloud( const PointCloud& pointCloud ) = 0;

    /// computes the closest points on point cloud to given points
    virtual Expected<void> findProjections( std::vector<PointsProjectionResult>& results,
        const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings ) const = 0;

    /// Returns amount of memory needed to compute projections
    virtual size_t projectionsHeapBytes( size_t numProjections ) const = 0;
};

/// default implementation of IPointsProjector
class MRMESH_CLASS PointsProjector : public IPointsProjector
{
public:
    /// sets the reference point cloud
    MRMESH_API Expected<void> setPointCloud( const PointCloud& pointCloud ) override;

    /// computes the closest points on point cloud to given points
    MRMESH_API Expected<void> findProjections( std::vector<PointsProjectionResult>& results,
        const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings ) const override;

    /// Returns amount of memory needed to compute projections
    MRMESH_API virtual size_t projectionsHeapBytes( size_t numProjections ) const override;
private:
    const PointCloud* pointCloud_{ nullptr };
};

/// \}
}