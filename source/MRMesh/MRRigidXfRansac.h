#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRAffineXf3.h"
#include "MRId.h"
#include <cstdint>
#include <vector>

namespace MR
{

struct RigidXfRansacParams
{
    /// a match is an inlier of a transformation if it maps the source point within this distance from the target point
    float inlierDistance = 0;

    /// the maximal number of random samples of three matches to test
    int maxIterations = 100000;

    /// the search stops earlier when the probability to have tested at least one sample of three inliers reaches this value
    float confidence = 0.999f;

    /// a sample is skipped without fitting if the distances between its source points and between its target points
    /// differ more than this ratio, since a rigid motion keeps distances
    float edgeLengthRatio = 0.9f;

    /// seed of pseudo-random sampling; the result depends on it but not on the number of threads
    std::uint32_t seed = 0;
};

struct RigidXfRansacResult
{
    /// rigid transformation of source points to target points
    AffineXf3f xf;

    /// the matches mapped by the transformation within inlierDistance
    std::vector<VertPair> inliers;
};

/// \brief Finds the rigid transformation of source points to target points from given matches, many of which can be wrong:
/// tests the transformations through random samples of three matches and keeps the one with most inliers,
/// then refines it by least squares fitting on all its inliers
/// \param matches (source point, target point) pairs, e.g. from findFPFHMatches(...)
/// \return error if there are less than three matches, no consistent sample was found or progress returned false
/// \ingroup AligningTransformGroup
[[nodiscard]] MRMESH_API Expected<RigidXfRansacResult> findRigidXfRansac( const VertCoords & srcPoints, const VertCoords & tgtPoints,
    const std::vector<VertPair> & matches, const RigidXfRansacParams & params, const ProgressCallback & progress = {} );

} //namespace MR
