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

    /// maximum distance from the fitted points to the surface; fitBSplineSurfaceToMesh measures it in the mesh vertices only
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
    /// requires buildGrid() to be called after the last change of the surface; the result is a local minimum of the distance, which can be
    /// not the closest point if the surface is strongly curved (e.g. nearly closed, where also the extensions of the fitted part come close
    /// to each other), or if the grid is sparse in space relative to the curvature (the grid is regular in the parameters, not in space)
    [[nodiscard]] MRMESH_API Vector2d project( const Vector3d& p ) const;

    /// squared distance from given point to the surface point found by project( p )
    [[nodiscard]] MRMESH_API double distanceSq( const Vector3d& p ) const;
};

struct FitBSplineSurfaceToMeshSettings
{
    /// surfaces with more and more control points (from 6x4 to 40x8) are fitted until the maximum distance from the vertices is below this value;
    /// if it is not reached, then the surface with the smallest distance is returned
    double tolerance = 0;

    /// the vertices get initial parameters in [margin, 1 - margin]^2 (they can move a little during the fit), and the rest of the domain [0, 1]^2
    /// is a natural extension of the surface; must be in [0, 0.5)
    double margin = 0.06;

    /// see BSplineSurfaceFitSettings::smoothing
    double smoothing = BSplineSurfaceFitSettings{}.smoothing;
};

/// approximates a mesh part with disk topology by a bicubic B-spline surface passing near its vertices:
/// the vertices are parametrized by leastSquaresConformalMap rotated to its bounding rectangle of the smallest area with u along the longer side
/// (the triangles of the part are counterclockwise in the parameters, so the normals of the surface and of the part agree), then several times
/// the surface is fitted by fitBSplineSurface and the vertices are projected on it to improve their parameters;
/// additional points with small weights are taken inside the triangles much larger than the typical one;
/// the grid of the returned surface is built, and its maxError is the maximum distance from the vertices of the part;
/// if the part is not one disk (several components, or touching itself in a vertex) then the parametrization can be meaningless,
/// and a surface with large maxError is returned, so pass a disk or check maxError;
/// \return an error if the part cannot be parametrized, has fewer than 35 vertices (too few for 6x4 control points), the margin is wrong, or no fit succeeds
[[nodiscard]] MRMESH_API Expected<FittedBSplineSurface> fitBSplineSurfaceToMesh( const MeshPart& mp, const FitBSplineSurfaceToMeshSettings& settings = {} );

} //namespace MR
