#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRVector.h"
#include "MRBitSet.h"
#include <array>

namespace MR
{

/// number of bins in each of three angular histograms of FPFH descriptor
inline constexpr int FPFHBins = 11;

/// number of values in FPFH descriptor
inline constexpr int FPFHDescriptorSize = 3 * FPFHBins;

/// Fast Point Feature Histogram of a point: three concatenated histograms of FPFHBins bins each,
/// describing the angles between the normals of nearby points and the directions between them;
/// each histogram sums to 1 (or to 0 for a point without neighbours)
using FPFHDescriptor = std::array<float, FPFHDescriptorSize>;
using FPFHDescriptors = Vector<FPFHDescriptor, VertId>;

/// \brief Computes Fast Point Feature Histograms for all valid points of given point cloud,
/// which can be compared to find corresponding points in two clouds independently of their relative position;
/// the descriptors do not change under rigid motions and uniform scaling of the cloud (if radius is scaled as well)
/// \param radius of neighborhood to consider, usually about 5 times larger than the distance between neighbour points
/// \return error if the cloud has no normals or progress returned false;
/// the normals must be oriented consistently, e.g. by makeOrientedNormals(...)
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API Expected<FPFHDescriptors> computeFPFH( const PointCloud& pointCloud,
    float radius, const ProgressCallback & progress = {} );

/// \brief Computes for each of given points how unusual its descriptor is: |d - mu| / sigma,
/// where d is the distance of the point's descriptor from the mean descriptor of all given points,
/// and mu and sigma are the mean and the standard deviation of these distances;
/// deviations |d - mu| up to 1e-6, which come from float rounding of equal descriptors, give 0
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API VertScalars computeFPFHUniqueness( const FPFHDescriptors & descriptors, const VertBitSet & points );

/// \brief Finds the points with unique descriptors, which are less likely to be matched ambiguously:
/// the points whose distance from the mean descriptor of all given points is outside the interval mu +- beta * sigma,
/// where mu and sigma are the mean and the standard deviation of these distances, i.e. computeFPFHUniqueness(...) > beta
/// \param beta the half-width of the interval in standard deviations, usually 1..2
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API VertBitSet findUniqueFPFHPoints( const FPFHDescriptors & descriptors, const VertBitSet & points, float beta );

/// \brief Computes FPFH descriptors of all valid points of given point cloud and finds the points with unique descriptors among them
/// \param radius of neighborhood to consider in computeFPFH(...)
/// \param beta the half-width of the interval in standard deviations in findUniqueFPFHPoints(...), usually 1..2
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API Expected<VertBitSet> findUniqueFPFHPoints( const PointCloud& pointCloud,
    float radius, float beta, const ProgressCallback & progress = {} );

/// \brief Finds for every source point the target point with the closest descriptor (in Euclidean distance)
/// by comparing all pairs of points, which is fast enough for clouds of several thousand points
/// \param mutual if true, keeps only the pairs where the source point is also the closest one to the target point
/// \return (source point, target point) pairs ordered by source point, or error if progress returned false
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API Expected<std::vector<VertPair>> findFPFHMatches(
    const FPFHDescriptors & srcDescriptors, const VertBitSet & srcPoints,
    const FPFHDescriptors & tgtDescriptors, const VertBitSet & tgtPoints,
    bool mutual = true, const ProgressCallback & progress = {} );

} //namespace MR
