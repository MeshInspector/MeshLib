#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRVector.h"
#include <array>

namespace MR
{

/// number of bins in each of three angular histograms of FPFH descriptor
inline constexpr int FPFHBins = 11;

/// Fast Point Feature Histogram of a point: three concatenated histograms of FPFHBins bins each,
/// describing the angles between the normals of nearby points and the directions between them;
/// each histogram sums to 1 (or to 0 for a point without neighbours)
using FPFHDescriptor = std::array<float, 3 * FPFHBins>;
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

} //namespace MR
