#pragma once

#include "MRMeshFwd.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include <vector>

namespace MR
{

/// non-rational B-spline curve in 3D
/// \ingroup MathGroup
struct BSplineCurve
{
    int degree = 3;                      ///< from 1 to 7
    std::vector<double> knots;           ///< non-decreasing, controlPoints.size() + degree + 1 values
    std::vector<Vector3d> controlPoints;

    /// the point of the curve at parameter t from [knots[degree], knots[controlPoints.size()]]
    [[nodiscard]] MRMESH_API Vector3d eval( double t ) const;
};

/// non-rational tensor product B-spline surface
/// \ingroup MathGroup
struct BSplineSurface
{
    int degreeU = 3;                     ///< from 1 to 7
    int degreeV = 3;                     ///< from 1 to 7
    std::vector<double> knotsU;          ///< non-decreasing, numU() + degreeU + 1 values
    std::vector<double> knotsV;          ///< non-decreasing, numV() + degreeV + 1 values
    std::vector<Vector3d> controlPoints; ///< numU() * numV() points, the point (i, j) has index i * numV() + j

    /// the number of control points along u and v
    [[nodiscard]] int numU() const { return int( knotsU.size() ) - degreeU - 1; }
    [[nodiscard]] int numV() const { return int( knotsV.size() ) - degreeV - 1; }

    /// the point of the surface at given parameters, and optionally its partial derivatives
    [[nodiscard]] MRMESH_API Vector3d eval( const Vector2d& uv, Vector3d* du = nullptr, Vector3d* dv = nullptr ) const;

    /// unit normal of the surface at given parameters: cross( du, dv ) normalized
    [[nodiscard]] MRMESH_API Vector3d normal( const Vector2d& uv ) const;

    /// parameters of the surface point closest to given one, found by Newton iterations from the initial parameters uv and kept in the domain
    [[nodiscard]] MRMESH_API Vector2d project( const Vector3d& p, Vector2d uv, int iterations = 12 ) const;
};

/// clamped knots with uniform inner spacing on [0, 1] for given number of control points and degree
[[nodiscard]] MRMESH_API std::vector<double> makeClampedUniformKnots( int numControlPoints, int degree );

/// B-spline curve of degree min( 3, points.size() - 1 ) passing through all given points (at least 2),
/// with the parameters of the points proportional to the length of the polyline through them, in [0, 1]
[[nodiscard]] MRMESH_API BSplineCurve interpolateBSplineCurve( const std::vector<Vector3d>& points );

struct BSplineSurfaceFitSettings
{
    /// the number of control points along u and v, at least 4 (the degree is 3)
    int numU = 4;
    int numV = 4;

    /// the weight of the second differences of the control net relative to the points, to keep the surface smooth where the points are sparse
    double smoothing = 1e-7;

    /// optional weights of the points, all 1 if not given
    const std::vector<double>* weights = nullptr;
};

/// bicubic B-spline surface over [0, 1] x [0, 1] fitted by least squares to the points with given parameters:
/// minimizes the sum of weight * squared distance between each point and the surface at its parameters
[[nodiscard]] MRMESH_API BSplineSurface fitBSplineSurface( const std::vector<Vector3d>& points, const std::vector<Vector2d>& uvs,
    const BSplineSurfaceFitSettings& settings = {} );

} //namespace MR
