#include "MRMeshToStepSurf.hpp"
#include "MRBestFit.h"
#include "MRMatrix3.h"
#include "MRMatrix4.h"
#include "MRSymMatrix3.h"
#include "MRSymMatrix4.h"
#include "MRVector4.h"

#include "MRToFromEigen.h"
#include <Eigen/Dense>

namespace MR
{

namespace MeshToStep
{

V3 centroid( const std::vector<V3>& pts )
{
    V3 res;
    for ( const auto& p : pts )
        res += p;
    return pts.empty() ? res : res / double( pts.size() );
}

bool fitCircle2( const std::vector<double>& xs, const std::vector<double>& ys, double& cx, double& cy, double& rad )
{
    double mx = 0, my = 0;
    for ( size_t i = 0; i < xs.size(); ++i )
    {
        mx += xs[i];
        my += ys[i];
    }
    mx /= std::max<size_t>( xs.size(), 1 );
    my /= std::max<size_t>( xs.size(), 1 );
    SymMatrix3d m;
    V3 rhs;
    for ( size_t i = 0; i < xs.size(); ++i )
    {
        const double x = xs[i] - mx, y = ys[i] - my;
        const V3 row( 2 * x, 2 * y, 1 );
        m += outerSquare( row );
        rhs += row * ( x * x + y * y );
    }
    // the rank-revealing solver gives the stablest result for a short arc, where the matrix is nearly degenerate
    const V3 sol = fromEigen( Eigen::Vector3d( toEigen( m ).completeOrthogonalDecomposition().solve( toEigen( rhs ) ) ) );
    const double r2 = sol.z + sol.x * sol.x + sol.y * sol.y;
    if ( !( r2 > 0 ) || !std::isfinite( r2 ) )
        return false;
    cx = sol.x + mx;
    cy = sol.y + my;
    rad = std::sqrt( r2 );
    return true;
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
    const V3 mean = centroid( pts );
    SymMatrix4d m;
    Vector4d rhs;
    for ( const auto& p : pts )
    {
        const V3 d = p - mean;
        const Vector4d row( 2 * d.x, 2 * d.y, 2 * d.z, 1 );
        m += outerSquare( row );
        rhs += row * d.lengthSq();
    }
    const Vector4d sol = m.pseudoinverse() * rhs;
    const V3 c( sol.x, sol.y, sol.z );
    const double r2 = sol.w + c.lengthSq();
    if ( !( r2 > 0 ) || !std::isfinite( r2 ) )
        return {};
    Surf res;
    res.type = SurfType::Sphere;
    res.c = mean + c;
    res.r = std::sqrt( r2 );
    return res;
}

std::optional<Surf> fitCyl( const std::vector<V3>& pts, const std::vector<V3>& normals )
{
    SymMatrix3d m;
    for ( const auto& n : normals )
        m += outerSquare( n );
    Matrix3d vecs;
    m.eigens( &vecs );
    Surf res;
    res.type = SurfType::Cyl;
    res.a = vecs.x;
    const auto [u, v] = res.a.perpendicular();
    std::vector<double> xs( pts.size() ), ys( pts.size() );
    for ( size_t i = 0; i < pts.size(); ++i )
    {
        xs[i] = dot( pts[i], u );
        ys[i] = dot( pts[i], v );
    }
    double cx, cy;
    if ( !fitCircle2( xs, ys, cx, cy, res.r ) )
        return {};
    res.c = cx * u + cy * v;
    return res;
}

bool revolutionAxis( const std::vector<V3>& pts, const std::vector<V3>& normals, V3& dir, V3& pos )
{
    double sc = 1e-9;
    for ( const auto& p : pts )
        sc = std::max( { sc, std::abs( p.x ), std::abs( p.y ), std::abs( p.z ) } );
    Eigen::Matrix<double, 6, 6> m = Eigen::Matrix<double, 6, 6>::Zero();
    for ( size_t i = 0; i < pts.size(); ++i )
    {
        const V3 mo = cross( pts[i] / sc, normals[i] );
        Eigen::Matrix<double, 6, 1> row;
        row << mo.x, mo.y, mo.z, normals[i].x, normals[i].y, normals[i].z;
        m += row * row.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es( m );
    const auto v = es.eigenvectors().col( 0 );
    V3 d( v[0], v[1], v[2] ), mom( v[3], v[4], v[5] );
    const double nd = d.length();
    if ( !( nd > 1e-6 ) )
        return false;
    d /= nd;
    mom /= nd;
    mom -= d * dot( mom, d );
    dir = d;
    pos = cross( d, mom ) * sc;
    return true;
}

std::optional<Surf> axisCone( const std::vector<V3>& pts, const V3& a, const V3& c, double* maxErr )
{
    double sh = 0, shh = 0, sr = 0, shr = 0;
    for ( const auto& p : pts )
    {
        const V3 d = p - c;
        const double h = dot( d, a );
        const double rho = ( d - h * a ).length();
        sh += h;
        shh += h * h;
        sr += rho;
        shr += h * rho;
    }
    const double n = double( pts.size() );
    const double det = shh * n - sh * sh;
    if ( std::abs( det ) < 1e-30 )
        return {};
    const double k = ( shr * n - sh * sr ) / det;
    const double b = ( shh * sr - sh * shr ) / det;
    if ( maxErr )
    {
        *maxErr = 0;
        for ( const auto& p : pts )
        {
            const V3 d = p - c;
            const double h = dot( d, a );
            *maxErr = std::max( *maxErr, std::abs( ( d - h * a ).length() - k * h - b ) );
        }
    }
    if ( std::abs( k ) < 1e-6 )
        return {};
    Surf res;
    res.type = SurfType::Cone;
    res.c = c + a * ( -b / k );
    res.a = k > 0 ? a : -a;
    res.r = std::atan( std::abs( k ) );
    return res;
}

std::optional<Surf> axisTorus( const std::vector<V3>& pts, const V3& a, const V3& c )
{
    std::vector<double> rhos( pts.size() ), hs( pts.size() );
    for ( size_t i = 0; i < pts.size(); ++i )
    {
        const V3 d = pts[i] - c;
        hs[i] = dot( d, a );
        rhos[i] = ( d - hs[i] * a ).length();
    }
    double cr, ch, rad;
    if ( !fitCircle2( rhos, hs, cr, ch, rad ) || cr <= 0 )
        return {};
    Surf res;
    res.type = SurfType::Torus;
    res.c = c + a * ch;
    res.a = a;
    res.R = cr;
    res.r = rad;
    return res;
}

std::optional<Surf> fitConeByNormals( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals )
{
    // all normals make the same angle with the axis: dot( n, a ) = k
    SymMatrix4d m;
    SymMatrix3d nn;
    V3 rhs;
    for ( size_t i = 0; i < normals.size(); ++i )
    {
        const auto& n = normals[i];
        m += outerSquare( Vector4d( n.x, n.y, n.z, -1 ) );
        nn += outerSquare( n );
        rhs += n * dot( n, pc[i] );
    }
    Matrix4d vecs;
    m.eigens( &vecs );
    V3 a( vecs.x.x, vecs.x.y, vecs.x.z );
    const double len = a.length();
    if ( !( len > 1e-9 ) )
        return {};
    a /= len;
    // the apex is on all tangent planes
    Surf res;
    res.type = SurfType::Cone;
    res.c = nn.pseudoinverse() * rhs;
    double sumH = 0, sumHH = 0, sumHR = 0;
    for ( const auto& p : pts )
    {
        const V3 d = p - res.c;
        const double h = dot( d, a );
        sumH += h;
        sumHH += h * h;
        sumHR += std::abs( h ) * ( d - h * a ).length();
    }
    if ( !( sumHH > 0 ) )
        return {};
    res.a = sumH < 0 ? -a : a;
    res.r = std::atan( sumHR / sumHH );
    return res.finite() && res.r > 1e-6 ? res : std::optional<Surf>();
}

std::optional<Surf> fitRevolution( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals, SurfType type )
{
    V3 a, c;
    if ( !revolutionAxis( pc, normals, a, c ) )
        return {};
    return type == SurfType::Cone ? axisCone( pts, a, c ) : axisTorus( pts, a, c );
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
    const int np = s0.type == SurfType::Torus ? 8 : 7;
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
        else if ( s.type == SurfType::Cone )
            s.r = x[6];
        else
        {
            s.R = x[6];
            s.r = std::abs( x[7] );
        }
        return s;
    };
    auto residuals = [&]( const Eigen::VectorXd& x, Eigen::VectorXd& f )
    {
        const Surf s = unpack( x );
        for ( size_t i = 0; i < pts.size(); ++i )
            f[i] = s.dist( pts[i] );
    };
    Eigen::VectorXd x( np );
    x.head( 7 ) << s0.c.x, s0.c.y, s0.c.z, s0.a.x, s0.a.y, s0.a.z, s0.r;
    if ( s0.type == SurfType::Torus )
    {
        x[6] = s0.R;
        x[7] = s0.r;
    }
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
