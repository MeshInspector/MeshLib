#pragma once

#include "MRMeshFwd.h"
#include "MRBitSet.h"

namespace MR
{

struct MarkedContour3f
{
    Contour3f contour;
    BitSet marks; ///< subset of indices of contour field
};

/// \return marked contour with all points from (in) marked
[[nodiscard]] inline MarkedContour3f markedContour( Contour3f in )
{
    const auto sz = in.size();
    return MarkedContour3f{ .contour = std::move( in ), .marks = BitSet( sz, true ) };
}

/// \return marked contour with only first and last points from (in) marked
[[nodiscard]] inline MarkedContour3f markedFirstLast( Contour3f in )
{
    const auto sz = in.size();
    MarkedContour3f res{ .contour = std::move( in ), .marks = BitSet( sz, false ) };
    res.marks.set( 0 );
    res.marks.set( sz - 1 );
    return res;
}

/// \param in input marked contour
/// \param maxStep maximum distance from not-marked point of returned contour to a neighbor point
/// \return contour with same marked points and not-marked points located on input contour with given maximum distance along it
[[nodiscard]] MRMESH_API MarkedContour3f resample( const MarkedContour3f & in, float maxStep );

/// \param in input marked contour
/// \param markStability a positive value, the more the value the closer marked points will be to their original positions
/// \return contour with same number of points and same marked, where each return point tries to be on a smooth curve
[[nodiscard]] MRMESH_API MarkedContour3f makeSpline( MarkedContour3f in, float markStability = 1 );

struct SplineSettings
{
    /// additional points will be added between each pair of control points,
    /// until the distance between neighbor points becomes less than this distance
    float samplingStep = 1;

    /// a positive value, the more the value the closer resulting spline will be to given control points
    float controlStability = 1;
};

/// \param controlPoints ordered point the spline to interpolate
/// \return spline contour with same or more points than initially given, marks field highlights the points corresponding to input ones
[[nodiscard]] MRMESH_API MarkedContour3f makeSpline( const Contour3f & controlPoints, const SplineSettings & settings );

} //namespace MR
