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

    /// if both are set, a transformation is scored by how many of these source points it moves within inlierDistance
    /// of some point of the target cloud, instead of by the number of its inlier matches;
    /// this finds the transformation even when only a few matches are correct, at the cost of a closest point search per point
    const VertCoords * overlapSrcPoints = nullptr;
    const PointCloud * overlapTgtCloud = nullptr;
};

/// counts the matches (source point, target point) that given transformation maps within given distance from their target points
/// \ingroup AligningTransformGroup
[[nodiscard]] MRMESH_API size_t countInliers( const VertCoords & srcPoints, const VertCoords & tgtPoints,
    const std::vector<VertPair> & matches, const AffineXf3f & xf, float inlierDistance );

/// \brief Finds the rigid transformation of source points to target points from given matches, many of which can be wrong:
/// tests the transformations through random samples of three matches and keeps the one with most inliers
/// (or the largest overlap, see RigidXfRansacParams::overlapSrcPoints), then refines it by least squares fitting on all its inlier matches
/// \param matches (source point, target point) pairs, e.g. from findFPFHMatches(...)
/// \return rigid transformation of source points to target points,
/// or error if there are less than three matches, no consistent sample was found or progress returned false
/// \ingroup AligningTransformGroup
[[nodiscard]] MRMESH_API Expected<AffineXf3f> findRigidXfRansac( const VertCoords & srcPoints, const VertCoords & tgtPoints,
    const std::vector<VertPair> & matches, const RigidXfRansacParams & params, const ProgressCallback & progress = {} );

} //namespace MR
