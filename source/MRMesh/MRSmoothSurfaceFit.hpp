#pragma once

#include "MRMeshFwd.h"
#include "MRConstants.h"
#include "MRFittedBSplineSurface.h"
#include "MRVector.h"
#include "MRVector2.h"
#include "MRVector3.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>

namespace MR
{

namespace SmoothSurfaceFit
{

using V3 = Vector3d;
using V2 = Vector2d;

/// the order of types is from the simplest to the most complex
enum class SurfType { Plane, Cyl, Cone, Sphere, Torus, BSpline };

struct Surf
{
    SurfType type = SurfType::Plane;
    V3 c;             ///< plane: a point, cylinder: axis point closest to the origin, cone: apex, sphere and torus: center
    V3 a{ 0, 0, 1 };  ///< unit normal of plane or axis direction
    double r = 0;     ///< radius of cylinder or sphere, minor radius of torus, half-angle of cone
    double R = 0;     ///< major radius of torus
    std::shared_ptr<const FittedBSplineSurface> bs;

    /// signed distance to the surface
    double dist( const V3& p ) const
    {
        if ( type == SurfType::Plane )
            return dot( p - c, a );
        if ( type == SurfType::Sphere )
            return ( p - c ).length() - r;
        if ( type == SurfType::BSpline )
        {
            const auto uv = bs->project( p );
            return dot( p - bs->surface.eval( uv ), bsplineNormal_( uv ) );
        }
        const V3 d = p - c;
        const double h = dot( d, a );
        const double rho = ( d - h * a ).length();
        if ( type == SurfType::Cyl )
            return rho - r;
        if ( type == SurfType::Cone )
            return ( rho - h * std::tan( r ) ) * std::cos( r );
        return std::hypot( rho - R, h ) - r;
    }

    /// unit normal of the surface in the point closest to given one (directed outside for closed surfaces)
    V3 normal( const V3& p ) const
    {
        if ( type == SurfType::Plane )
            return a;
        if ( type == SurfType::BSpline )
            return bsplineNormal_( bs->project( p ) );
        const V3 d = p - c;
        if ( type == SurfType::Sphere )
            return d / std::max( d.length(), 1e-12 );
        const double h = dot( d, a );
        const V3 rv = d - h * a;
        const double rho = rv.length();
        const V3 er = rv / std::max( rho, 1e-12 );
        if ( type == SurfType::Cyl )
            return er;
        if ( type == SurfType::Cone )
            return er * std::cos( r ) - a * std::sin( r );
        const V3 n = er * ( rho - R ) + h * a;
        return n / std::max( n.length(), 1e-12 );
    }

    /// unit normal of the B-spline surface, or (0,0,1) where it is not defined
    V3 bsplineNormal_( const V2& uv ) const
    {
        const V3 n = bs->surface.normal( uv );
        return n == V3() ? V3( 0, 0, 1 ) : n;
    }

    bool finite() const
    {
        return std::isfinite( c.x + c.y + c.z + a.x + a.y + a.z + r + R );
    }

    double maxDist( const Vector<V3>& pts ) const
    {
        double res = 0;
        for ( const auto& p : pts )
            res = std::max( res, std::abs( dist( p ) ) );
        return res;
    }
};

V3 centroid( const Vector<V3>& pts );

Surf fitPlane( const Vector<V3>& pts );

std::optional<Surf> fitSphere( const Vector<V3>& pts );

/// cylinder with the axis orthogonal to given normals
std::optional<Surf> fitCyl( const Vector<V3>& pts, const Vector<V3>& normals );

/// cone with given axis fitted to the points
std::optional<Surf> axisCone( const Vector<V3>& pts, const V3& a, const V3& c );

/// torus with given axis fitted to the points
std::optional<Surf> axisTorus( const Vector<V3>& pts, const V3& a, const V3& c );

/// cone tangent to the planes passing through the points pc with given normals; unlike fitRevolution, it works for one strip of triangles
std::optional<Surf> fitConeByNormals( const Vector<V3>& pts, const Vector<V3>& pc, const Vector<V3>& normals );

/// cone or torus fitted to the points, with the axis found from the normals in the points pc
std::optional<Surf> fitRevolution( const Vector<V3>& pts, const Vector<V3>& pc, const Vector<V3>& normals, SurfType type );

/// minimizes the sum of squared distances from the points to the surface by Levenberg-Marquardt method
Surf refine( const Surf& s0, const Vector<V3>& pts );

} //namespace SmoothSurfaceFit

} //namespace MR
