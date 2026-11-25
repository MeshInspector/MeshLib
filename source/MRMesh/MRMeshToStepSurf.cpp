#include "MRMeshToStepSurf.hpp"
#include "MRBestFit.h"
#include "MRLine3.h"
#include "MRMatrix3.h"
#include "MRPlane3.h"

#include <Eigen/Dense>

namespace MR
{

namespace MeshToStep
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

V3 centroid( const std::vector<V3>& pts )
{
    V3 res;
    for ( const auto& p : pts )
        res += p;
    return pts.empty() ? res : res / double( pts.size() );
}

Surf fitPlane( const std::vector<V3>& pts )
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

std::optional<Surf> fitSphere( const std::vector<V3>& pts )
{
    if ( const auto sphere = findBestSphere( pts ) )
        return toSurf( *sphere );
    return {};
}

std::optional<Surf> fitCyl( const std::vector<V3>& pts, const std::vector<V3>& normals )
{
    if ( const auto cyl = findBestCylinderByNormals( pts, normals ) )
        return toSurf( *cyl );
    return {};
}

std::optional<Surf> axisCone( const std::vector<V3>& pts, const V3& a, const V3& c )
{
    if ( const auto cone = findBestConeWithAxis( pts, Line3d( c, a ) ) )
        return toSurf( *cone );
    return {};
}

std::optional<Surf> axisTorus( const std::vector<V3>& pts, const V3& a, const V3& c )
{
    if ( const auto torus = findBestTorusWithAxis( pts, Line3d( c, a ) ) )
        return toSurf( *torus );
    return {};
}

std::optional<Surf> fitConeByNormals( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals )
{
    std::vector<Plane3d> planes;
    planes.reserve( normals.size() );
    for ( size_t i = 0; i < normals.size(); ++i )
        planes.push_back( Plane3d::fromDirAndPt( normals[i], pc[i] ) );
    if ( const auto cone = findBestConeByTangentPlanes( pts, planes ) )
        return toSurf( *cone );
    return {};
}

std::optional<Surf> fitRevolution( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals, SurfType type )
{
    std::vector<Line3d> normalLines;
    normalLines.reserve( normals.size() );
    for ( size_t i = 0; i < normals.size(); ++i )
        normalLines.emplace_back( pc[i], normals[i] );
    const auto axis = findRevolutionAxis( normalLines );
    if ( !axis )
        return {};
    return type == SurfType::Cone ? axisCone( pts, axis->d, axis->p ) : axisTorus( pts, axis->d, axis->p );
}

Surf refine( const Surf& s0, const std::vector<V3>& pts )
{
    if ( s0.type == SurfType::Plane )
        return fitPlane( pts );
    if ( s0.type == SurfType::Sphere )
    {
        auto res = fitSphere( pts );
        return res ? *res : s0;
    }
    if ( s0.type == SurfType::BSpline || pts.empty() )
        return s0;
    if ( s0.type == SurfType::Torus )
    {
        const Torus3d torus( s0.c, s0.a, s0.R, s0.r );
        const auto refined = refineTorus( pts, torus );
        return refined == torus ? s0 : toSurf( refined );
    }
    // cylinder or cone
    constexpr int np = 7;
    auto unpack = [&]( const Eigen::VectorXd& x )
    {
        Surf s;
        s.type = s0.type;
        s.c = V3( x[0], x[1], x[2] );
        s.a = V3( x[3], x[4], x[5] ).normalized();
        if ( s.type == SurfType::Cyl )
        {
            s.c -= s.a * dot( s.c, s.a );
            s.r = std::abs( x[6] );
        }
        else
            s.r = x[6];
        return s;
    };
    auto residuals = [&]( const Eigen::VectorXd& x, Eigen::VectorXd& f )
    {
        const Surf s = unpack( x );
        for ( size_t i = 0; i < pts.size(); ++i )
            f[i] = s.dist( pts[i] );
    };
    Eigen::VectorXd x( np );
    x << s0.c.x, s0.c.y, s0.c.z, s0.a.x, s0.a.y, s0.a.z, s0.r;
    const int m = int( pts.size() );
    Eigen::VectorXd f( m ), f1( m );
    Eigen::MatrixXd jac( m, np );
    residuals( x, f );
    double cost = f.squaredNorm();
    if ( !std::isfinite( cost ) )
        return s0;
    double lambda = 1e-3;
    for ( int it = 0; it < 60; ++it )
    {
        for ( int j = 0; j < np; ++j )
        {
            Eigen::VectorXd xh = x;
            const double h = 1e-7 * std::max( 1.0, std::abs( x[j] ) );
            xh[j] += h;
            residuals( xh, f1 );
            jac.col( j ) = ( f1 - f ) / h;
        }
        const Eigen::MatrixXd jtj = jac.transpose() * jac;
        const Eigen::VectorXd g = jac.transpose() * f;
        bool improved = false;
        for ( int k = 0; k < 12; ++k )
        {
            Eigen::MatrixXd lhs = jtj;
            for ( int j = 0; j < np; ++j )
                lhs( j, j ) += lambda * std::max( jtj( j, j ), 1e-12 );
            const Eigen::VectorXd dx = lhs.ldlt().solve( -g );
            const Eigen::VectorXd xn = x + dx;
            residuals( xn, f1 );
            const double c1 = f1.squaredNorm();
            if ( std::isfinite( c1 ) && c1 < cost )
            {
                const bool converged = cost - c1 <= 1e-14 * cost;
                x = xn;
                f = f1;
                cost = c1;
                lambda = std::max( lambda * 0.2, 1e-12 );
                improved = !converged;
                break;
            }
            lambda *= 5;
        }
        if ( !improved )
            break;
    }
    Surf res = unpack( x );
    if ( res.type == SurfType::Cone )
    {
        // half-angle must be in (0, pi/2)
        res.r = std::remainder( res.r, 2 * PI );
        if ( res.r < 0 )
        {
            res.r = -res.r;
            res.a = -res.a;
        }
        if ( res.r > PI / 2 )
        {
            res.r = PI - res.r;
            res.a = -res.a;
        }
    }
    return res.finite() ? res : s0;
}

bool sameSurface( const Surf& p, const Surf& q, double tol )
{
    if ( p.type != q.type || p.type == SurfType::BSpline )
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

bool isCoaxial( const Surf& s, const std::vector<Axis>& axes, double tol )
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

void addAxis( std::vector<Axis>& axes, const Surf& s, double tol )
{
    if ( !isCoaxial( s, axes, tol ) )
        axes.push_back( { s.a, s.c - s.a * dot( s.c, s.a ) } );
}

} //namespace MeshToStep

} //namespace MR
