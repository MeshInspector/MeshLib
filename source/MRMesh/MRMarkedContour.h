#pragma once

#include "MRBitSet.h"
#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

struct MarkedContour3f
{
    Contour3f contour;
    BitSet marks; ///< indices of control (marked) points
};

[[nodiscard]] inline bool isClosed( const Contour3f& c ) { return c.size() > 1 && c.front() == c.back(); }

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

/// keeps all marked points from input contour and adds/removes other points to have them as many as possible,
/// but at the distance along the input line not shorter than (minStep) from their neighbor points
/// \param normals optional parameter with the normals of input points that will be resampled to become normals of output points
[[nodiscard]] MRMESH_API MarkedContour3f resample( const MarkedContour3f & in, float minStep, Contour3f * normals = nullptr );

/// \param in input marked contour
/// \param markStability a positive value, the more the value the closer marked points will be to their original positions
/// \param normals if provided the curve at marked points will try to be orthogonal to given normal there
/// \return contour with same number of points and same marked, where each return point tries to be on a smooth curve
[[nodiscard]] MRMESH_API MarkedContour3f makeSpline( MarkedContour3f in, float markStability = 1, const Contour3f * normals = nullptr );

/// \param in input marked contour
/// \param normals the curve at marked points will try to be orthogonal to given normal there
/// \param markStability a positive value, the more the value the closer marked points will be to their original positions
/// \return contour with same number of points and same marked, where each return point tries to be on a smooth curve
[[nodiscard]] MRMESH_API MarkedContour3f makeSpline( MarkedContour3f in, const Contour3f & normals, float markStability = 1 );

struct SplineSettings
{
    /// additional points will be added between each pair of control points,
    /// until the distance between neighbor points becomes less than this distance
    float samplingStep = 1;

    /// a positive value, the more the value the closer resulting spline will be to given control points
    float controlStability = 1;

    /// the shape of resulting spline depends on the total number of points in the contour,
    /// which in turn depends on the length of input contour being sampled;
    /// setting iterations greater than one allows you to pass a constructed spline as a better input contour to the next run of the algorithm
    int iterations = 1;

    /// optional parameter with the normals of input points that will be resampled to become normals of output points
    Contour3f * normals = nullptr;

    /// if true and normals are provided, then the curve at marked points will try to be orthogonal to given normal there
    bool normalsAffectShape = false;
};

/// \param controlPoints ordered point the spline to interpolate
/// \return spline contour with same or more points than initially given, marks field highlights the points corresponding to input ones
[[nodiscard]] MRMESH_API MarkedContour3f makeSpline( const Contour3f & controlPoints, const SplineSettings & settings );

} //namespace MR
