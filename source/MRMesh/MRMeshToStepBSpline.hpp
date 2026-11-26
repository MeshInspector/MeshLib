#pragma once

#include "MRMeshFwd.h"
#include "MRBSpline.h"
#include "MRVector.h"
#include "MRVector2.h"
#include "MRVector3.h"

#include <array>
#include <memory>

namespace MR
{

namespace MeshToStep
{

using V3 = Vector3d;
using V2 = Vector2d;
using Tri = std::array<int, 3>;

/// bicubic B-spline surface over [0,1]x[0,1] fitted to a mesh region
struct BSplineSurf
{
    BSplineSurface surf;
    double err = 0;  ///< maximum distance from fitted points
    Vector<V3> grid; ///< surface points in a regular grid of parameters, where the projections start from

    void buildGrid();

    /// parameters of the closest surface point, found starting from the closest grid point
    V2 project( const V3& p ) const;

    /// unit normal of the surface, or (0,0,1) where it is not defined
    V3 normalAt( const V2& uv ) const;

    double absDist( const V3& p ) const { return ( p - surf.eval( project( p ) ) ).length(); }
};

/// approximates a mesh region with disk topology by a B-spline surface passing near its vertices;
/// \return nullptr if the region cannot be parametrized or has too few vertices
std::shared_ptr<BSplineSurf> fitBSpline( const Mesh& mesh, const FaceBitSet& region, double tol );

} //namespace MeshToStep

} //namespace MR
