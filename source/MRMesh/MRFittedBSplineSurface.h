#pragma once

#include "MRMeshFwd.h"
#include "MRBSpline.h"
#include "MRExpected.h"

namespace MR
{

/// B-spline surface fitted to some points, with the points of the surface in a regular grid of its parameters,
/// which allow projecting on the surface without initial parameters
/// \ingroup MathGroup
struct FittedBSplineSurface
{
    BSplineSurface surface;

    /// maximum distance from the fitted points to the surface
    double maxError = 0;

    /// the number of grid points along u and along v
    int gridSize = 0;

    /// gridSize * gridSize points of the surface, the point (i, j) with the parameters gridUV( i, j ) has index i * gridSize + j
    Vector<Vector3d> grid;

    /// computes the grid with given number of points along u and v, at least 2; the surface must be valid
    MRMESH_API void buildGrid( int size = 60 );

    /// the parameters of the grid point (i, j): they divide the domain of the surface in ( gridSize - 1 )^2 equal cells
    [[nodiscard]] MRMESH_API Vector2d gridUV( int i, int j ) const;

    /// parameters of the surface point closest to given one: BSplineSurface::project starting from the closest point of the grid;
    /// requires buildGrid() to be called after the last change of the surface
    [[nodiscard]] MRMESH_API Vector2d project( const Vector3d& p ) const;

    /// the distance from given point to the surface point found by project( p )
    [[nodiscard]] MRMESH_API double distance( const Vector3d& p ) const;
};

struct FitBSplineSurfaceToMeshSettings
{
    /// surfaces with more and more control points (from 6x4 to 40x8) are fitted until the maximum distance from the vertices is below this value;
    /// if it is not reached, then the surface with the smallest distance is returned
    double tolerance = 0;

    /// the vertices get the parameters in [margin, 1 - margin]^2, and the rest of the domain [0, 1]^2 is a natural extension of the surface
    double margin = 0.06;

    /// see BSplineSurfaceFitSettings::smoothing
    double smoothing = 1e-7;
};

/// approximates a mesh part with disk topology by a bicubic B-spline surface passing near its vertices:
/// the vertices are parametrized by leastSquaresConformalMap with u along the longest direction of the map, then several times
/// the surface is fitted by fitBSplineSurface and the vertices are projected on it to improve their parameters;
/// additional points with small weights are taken inside the triangles much larger than the typical one;
/// the grid of the returned surface is built, and its maxError is the maximum distance from the vertices of the part;
/// \return an error if the part cannot be parametrized, has fewer than 35 vertices (too few for 6x4 control points), or no fit succeeds
[[nodiscard]] MRMESH_API Expected<FittedBSplineSurface> fitBSplineSurfaceToMesh( const MeshPart& mp, const FitBSplineSurfaceToMeshSettings& settings = {} );

} //namespace MR
