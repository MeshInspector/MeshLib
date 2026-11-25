#pragma once

#include "MRMeshToStepBSpline.hpp"
#include "MRConstants.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace MR
{

namespace MeshToStep
{

const double cCos40 = std::cos( 40 * PI / 180 );
const double cCosVert = std::cos( 29.5 * PI / 180 );
const double cMaxFacetAngle = 35 * PI / 180;
const double cCosFacets = std::cos( cMaxFacetAngle );
const double cCos1 = std::cos( 1 * PI / 180 );

/// the order of types is from the simplest to the most complex
enum class SurfType { Plane, Cyl, Cone, Sphere, Torus, BSpline };

struct Surf
{
    SurfType type = SurfType::Plane;
    V3 c;             ///< plane: a point, cylinder: axis point closest to the origin, cone: apex, sphere and torus: center
    V3 a{ 0, 0, 1 };  ///< unit normal of plane or axis direction
    double r = 0;     ///< radius of cylinder or sphere, minor radius of torus, half-angle of cone
    double R = 0;     ///< major radius of torus
    bool known = false; ///< taken from already found surfaces, not to be refined
    std::shared_ptr<const BSplineSurf> bs;

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
            V3 s;
            bs->eval( uv, s );
            return dot( p - s, bs->normalAt( uv ) );
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
            return bs->normalAt( bs->project( p ) );
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

    bool finite() const
    {
        return std::isfinite( c.x + c.y + c.z + a.x + a.y + a.z + r + R );
    }

    double maxDist( const std::vector<V3>& pts ) const
    {
        double res = 0;
        for ( const auto& p : pts )
            res = std::max( res, std::abs( dist( p ) ) );
        return res;
    }
};

V3 centroid( const std::vector<V3>& pts );

/// least squares circle x^2 + y^2 = 2*cx*x + 2*cy*y + s; \return false if the points do not define a circle
bool fitCircle2( const std::vector<double>& xs, const std::vector<double>& ys, double& cx, double& cy, double& rad );

Surf fitPlane( const std::vector<V3>& pts );

std::optional<Surf> fitSphere( const std::vector<V3>& pts );

/// cylinder with the axis orthogonal to given normals
std::optional<Surf> fitCyl( const std::vector<V3>& pts, const std::vector<V3>& normals );

/// the line intersecting all normal lines: the axis of a surface of revolution
bool revolutionAxis( const std::vector<V3>& pts, const std::vector<V3>& normals, V3& dir, V3& pos );

/// cone with given axis fitted to the points
std::optional<Surf> axisCone( const std::vector<V3>& pts, const V3& a, const V3& c, double* maxErr = nullptr );

/// torus with given axis fitted to the points
std::optional<Surf> axisTorus( const std::vector<V3>& pts, const V3& a, const V3& c );

/// cone tangent to the planes passing through the points pc with given normals; unlike fitRevolution, it works for one strip of triangles
std::optional<Surf> fitConeByNormals( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals );

/// cone or torus fitted to the points, with the axis found from the normals in the points pc
std::optional<Surf> fitRevolution( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals, SurfType type );

/// minimizes the sum of squared distances from the points to the surface by Levenberg-Marquardt method
Surf refine( const Surf& s0, const std::vector<V3>& pts );

struct Axis
{
    V3 dir, pos;
};

/// \return true if two surfaces of the same analytic type coincide within the tolerance
bool sameSurface( const Surf& p, const Surf& q, double tol );

/// \return true if the axis of the surface coincides with one of given axes
bool isCoaxial( const Surf& s, const std::vector<Axis>& axes, double tol );

void addAxis( std::vector<Axis>& axes, const Surf& s, double tol );

} //namespace MeshToStep

} //namespace MR
