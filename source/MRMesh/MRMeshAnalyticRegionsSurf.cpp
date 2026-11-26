#include "MRMeshAnalyticRegionsSurf.hpp"
#include "MRBestFit.h"
#include "MRLine3.h"
#include "MRMatrix3.h"
#include "MRPlane3.h"

namespace MR
{

namespace AnalyticRegions
{

namespace
{

Surf toSurf( const Sphere3d& sphere )
{
    Surf res;
    res.type = SurfType::Sphere;
    res.c = sphere.center;
    res.r = sphere.radius;
    return res;
}

Surf toSurf( const Cylinder3d& cyl )
{
    Surf res;
    res.type = SurfType::Cyl;
    res.a = cyl.direction();
    res.c = cyl.center() - res.a * dot( cyl.center(), res.a );
    res.r = cyl.radius;
    return res;
}

Surf toSurf( const Cone3d& cone )
{
    Surf res;
    res.type = SurfType::Cone;
    res.c = cone.center();
    res.a = cone.direction();
    res.r = cone.angle;
    return res;
}

Surf toSurf( const Torus3d& torus )
{
    Surf res;
    res.type = SurfType::Torus;
    res.c = torus.center;
    res.a = torus.axis;
    res.R = torus.majorRadius;
    res.r = torus.minorRadius;
    return res;
}

} //anonymous namespace

V3 centroid( const Vector<V3>& pts )
{
    V3 res;
    for ( const auto& p : pts )
        res += p;
    return pts.empty() ? res : res / double( pts.size() );
}

Surf fitPlane( const Vector<V3>& pts )
{
    Surf res;
    PointAccumulator acc;
    for ( const auto& p : pts )
        acc.addPoint( p );
    Matrix3d vecs;
    V3 vals;
    if ( acc.getCenteredCovarianceEigen( res.c, vecs, vals ) )
        res.a = vecs.x;
    return res;
}

std::optional<Surf> fitSphere( const Vector<V3>& pts )
{
    if ( const auto sphere = findBestSphere( pts ) )
        return toSurf( *sphere );
    return {};
}

std::optional<Surf> fitCyl( const Vector<V3>& pts, const Vector<V3>& normals )
{
    if ( const auto cyl = findBestCylinderByNormals( pts, normals ) )
        return toSurf( *cyl );
    return {};
}

std::optional<Surf> axisCone( const Vector<V3>& pts, const V3& a, const V3& c )
{
    if ( const auto cone = findBestConeWithAxis( pts, Line3d( c, a ) ) )
        return toSurf( *cone );
    return {};
}

std::optional<Surf> axisTorus( const Vector<V3>& pts, const V3& a, const V3& c )
{
    if ( const auto torus = findBestTorusWithAxis( pts, Line3d( c, a ) ) )
        return toSurf( *torus );
    return {};
}

std::optional<Surf> fitConeByNormals( const Vector<V3>& pts, const Vector<V3>& pc, const Vector<V3>& normals )
{
    Vector<Plane3d> planes;
    planes.reserve( normals.size() );
    for ( size_t i = 0; i < normals.size(); ++i )
        planes.push_back( Plane3d::fromDirAndPt( normals[i], pc[i] ) );
    if ( const auto cone = findBestConeByTangentPlanes( pts, planes ) )
        return toSurf( *cone );
    return {};
}

std::optional<Surf> fitRevolution( const Vector<V3>& pts, const Vector<V3>& pc, const Vector<V3>& normals, SurfType type )
{
    Vector<Line3d> normalLines;
    normalLines.reserve( normals.size() );
    for ( size_t i = 0; i < normals.size(); ++i )
        normalLines.emplace_back( pc[i], normals[i] );
    const auto axis = findRevolutionAxis( normalLines );
    if ( !axis )
        return {};
    return type == SurfType::Cone ? axisCone( pts, axis->d, axis->p ) : axisTorus( pts, axis->d, axis->p );
}

Surf refine( const Surf& s0, const Vector<V3>& pts )
{
    if ( s0.type == SurfType::Plane )
        return fitPlane( pts );
    if ( s0.type == SurfType::Sphere )
    {
        auto res = fitSphere( pts );
        return res ? *res : s0;
    }
    if ( pts.empty() )
        return s0;
    if ( s0.type == SurfType::Torus )
    {
        const Torus3d torus( s0.c, s0.a, s0.R, s0.r );
        const auto refined = refineTorus( pts, torus );
        // no major radius means that a sphere fits the points better
        if ( refined == torus )
            return s0;
        // without major radius it is a sphere
        if ( refined.majorRadius < 1e-3 * refined.minorRadius )
            return toSurf( Sphere3d( refined.center, refined.minorRadius ) );
        return toSurf( refined );
    }
    // cylinder or cone; the refinements need at least 7 points, and a surface through fewer points (e.g. a cylinder through
    // two triangles) is found by them too if all the points are repeated: it does not change the closest surface
    Vector<V3> repeated;
    while ( pts.size() < 7 && repeated.size() < 7 )
        for ( const auto& q : pts )
            repeated.push_back( q );
    const auto& ps = repeated.empty() ? pts : repeated;
    const Surf res = s0.type == SurfType::Cyl ? toSurf( refineCylinder( ps, Cylinder3d( s0.c, s0.a, s0.r, 0 ) ) )
        : toSurf( refineCone( ps, Cone3d( Line3d( s0.c, s0.a ), s0.r, 0 ) ) );
    return res.finite() ? res : s0;
}

bool sameSurface( const Surf& p, const Surf& q, double tol )
{
    if ( p.type != q.type )
        return false;
    if ( p.type == SurfType::Sphere )
        return ( p.c - q.c ).length() < tol && std::abs( p.r - q.r ) < tol;
    if ( std::abs( std::abs( dot( p.a, q.a ) ) - 1 ) > 1e-7 )
        return false;
    const V3 d = q.c - p.c;
    switch ( p.type )
    {
    case SurfType::Plane:
        return std::abs( dot( d, p.a ) ) < tol;
    case SurfType::Cyl:
        return ( d - p.a * dot( d, p.a ) ).length() < tol && std::abs( p.r - q.r ) < tol;
    case SurfType::Cone:
        return d.length() < tol && std::abs( p.r - q.r ) < 1e-5 && dot( p.a, q.a ) > 0;
    default:
        return d.length() < tol && std::abs( p.r - q.r ) < tol && std::abs( p.R - q.R ) < tol;
    }
}

bool isCoaxial( const Surf& s, const Vector<Axis>& axes, double tol )
{
    for ( const auto& ax : axes )
    {
        if ( std::abs( std::abs( dot( ax.dir, s.a ) ) - 1 ) >= 1e-6 )
            continue;
        const V3 d = s.c - ax.pos;
        if ( ( d - ax.dir * dot( d, ax.dir ) ).length() < 5 * tol )
            return true;
    }
    return false;
}

void addAxis( Vector<Axis>& axes, const Surf& s, double tol )
{
    if ( !isCoaxial( s, axes, tol ) )
        axes.push_back( { s.a, s.c - s.a * dot( s.c, s.a ) } );
}

} //namespace AnalyticRegions

} //namespace MR
