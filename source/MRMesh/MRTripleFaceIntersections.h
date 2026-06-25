#pragma once

#include "MRIntersectionContour.h"
#include <compare>

namespace MR
{

/// a triple of faces
struct FaceFaceFace
{
    FaceId aFace;
    FaceId bFace;
    FaceId cFace;
    FaceFaceFace( FaceId a, FaceId b, FaceId c ) : aFace( a ), bFace( b ), cFace( c ) {}
    FaceFaceFace() {};
    auto operator<=>( const FaceFaceFace& rhs ) const = default;
};

/// given all self-intersection contours on a mesh, finds all intersecting triangle triples (every two triangles from a triple intersect);
/// please note that not all such triples will have a common point
[[nodiscard]] MRMESH_API std::vector<FaceFaceFace> findTripleFaceIntersections( const MeshTopology& topology, const ContinuousContours& selfContours );

} // namespace MR
