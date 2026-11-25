#include "MRMeshToStep.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRBox.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRSuppressWarning.h"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>

namespace MR
{

namespace
{

using V3 = Vector3d;
using V2 = Vector2d;

const double cCos40 = std::cos( 40 * PI / 180 );
const double cCosVert = std::cos( 29.5 * PI / 180 );
const double cCosFacets = std::cos( 35 * PI / 180 );
const double cCos1 = std::cos( 1 * PI / 180 );

/// values of (deg+1) non-zero B-spline basis functions at t, and optionally their derivatives;
/// \return the index of the first of them
int bsplineBasis( const std::vector<double>& knots, int deg, int n, double t, double* vals, double* ders )
{
    int span = deg;
    while ( span + 1 < n && knots[span + 1] <= t )
        ++span;
    double left[8], right[8], low[8];
    vals[0] = 1;
    for ( int j = 1; j <= deg; ++j )
    {
        if ( ders && j == deg )
            std::copy( vals, vals + deg, low );
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        double saved = 0;
        for ( int r = 0; r < j; ++r )
        {
            const double temp = vals[r] / ( right[r + 1] + left[j - r] );
            vals[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        vals[j] = saved;
    }
    if ( ders )
    {
        for ( int j = 0; j <= deg; ++j )
        {
            const int i = span - deg + j;
            const double t1 = j >= 1 ? low[j - 1] / ( knots[i + deg] - knots[i] ) : 0;
            const double t2 = j < deg ? low[j] / ( knots[i + deg + 1] - knots[i + 1] ) : 0;
            ders[j] = deg * ( t1 - t2 );
        }
    }
    return span - deg;
}

std::vector<double> clampedKnots( int n, int deg )
{
    std::vector<double> res( n + deg + 1, 1.0 );
    for ( int i = 0; i <= deg; ++i )
        res[i] = 0;
    for ( int i = deg + 1; i < n; ++i )
        res[i] = double( i - deg ) / ( n - deg );
    return res;
}

/// bicubic B-spline surface over [0,1]x[0,1]
struct BSplineSurf
{
    static constexpr int deg = 3;
    static constexpr int cGrid = 60;
    int nu = 0, nv = 0;
    std::vector<double> tu, tv;
    std::vector<V3> cp; ///< nu * nv points, [i * nv + j]
    double err = 0;     ///< maximum distance from fitted points
    std::vector<V3> grid;

    void eval( const V2& uv, V3& s, V3* su = nullptr, V3* sv = nullptr ) const
    {
        double bu[4], bv[4], du[4], dv[4];
        const int iu = bsplineBasis( tu, deg, nu, uv.x, bu, su ? du : nullptr );
        const int iv = bsplineBasis( tv, deg, nv, uv.y, bv, sv ? dv : nullptr );
        s = V3();
        if ( su )
            *su = V3();
        if ( sv )
            *sv = V3();
        for ( int a = 0; a <= deg; ++a )
            for ( int b = 0; b <= deg; ++b )
            {
                const V3& p = cp[( iu + a ) * nv + iv + b];
                s += bu[a] * bv[b] * p;
                if ( su )
                    *su += du[a] * bv[b] * p;
                if ( sv )
                    *sv += bu[a] * dv[b] * p;
            }
    }

    void buildGrid()
    {
        grid.resize( cGrid * cGrid );
        for ( int i = 0; i < cGrid; ++i )
            for ( int j = 0; j < cGrid; ++j )
                eval( gridUV( i * cGrid + j ), grid[i * cGrid + j] );
    }

    static V2 gridUV( int k ) { return V2( double( k / cGrid ) / ( cGrid - 1 ), double( k % cGrid ) / ( cGrid - 1 ) ); }

    /// parameters of the closest surface point; if no initial guess is given then starts from the closest grid sample
    V2 project( const V3& p, const V2* init = nullptr, int iters = 12 ) const
    {
        V2 uv;
        if ( init )
            uv = *init;
        else
        {
            int best = 0;
            double bestD = DBL_MAX;
            for ( int k = 0; k < (int)grid.size(); ++k )
                if ( auto d = ( grid[k] - p ).lengthSq(); d < bestD )
                {
                    bestD = d;
                    best = k;
                }
            uv = gridUV( best );
        }
        for ( int it = 0; it < iters; ++it )
        {
            V3 s, su, sv;
            eval( uv, s, &su, &sv );
            const V3 r = s - p;
            const double a = dot( su, su ), b = dot( su, sv ), c = dot( sv, sv ), e = -dot( su, r ), f = -dot( sv, r );
            double det = a * c - b * b;
            if ( std::abs( det ) < 1e-30 )
                det = 1e-30;
            uv.x = std::clamp( uv.x + ( c * e - b * f ) / det, 0.0, 1.0 );
            uv.y = std::clamp( uv.y + ( a * f - b * e ) / det, 0.0, 1.0 );
        }
        return uv;
    }

    V3 normalAt( const V2& uv ) const
    {
        V3 s, su, sv;
        eval( uv, s, &su, &sv );
        const V3 n = cross( su, sv );
        const double l = n.length();
        return l > 1e-30 ? n / l : V3( 0, 0, 1 );
    }

    double absDist( const V3& p ) const
    {
        V3 s;
        eval( project( p ), s );
        return ( p - s ).length();
    }
};

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

/// two unit vectors orthogonal to given unit vector and to each other
void frame( const V3& a, V3& u, V3& v )
{
    V3 t( 1, 0, 0 );
    if ( std::abs( a.y ) < std::abs( a.x ) && std::abs( a.y ) <= std::abs( a.z ) )
        t = V3( 0, 1, 0 );
    else if ( std::abs( a.z ) < std::abs( a.x ) && std::abs( a.z ) < std::abs( a.y ) )
        t = V3( 0, 0, 1 );
    u = cross( a, t ).normalized();
    v = cross( a, u );
}

V3 centroid( const std::vector<V3>& pts )
{
    V3 res;
    for ( const auto& p : pts )
        res += p;
    return pts.empty() ? res : res / double( pts.size() );
}

/// least squares circle x^2 + y^2 = 2*cx*x + 2*cy*y + s; \return false if the points do not define a circle
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
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for ( size_t i = 0; i < xs.size(); ++i )
    {
        const double x = xs[i] - mx, y = ys[i] - my;
        const Eigen::Vector3d row( 2 * x, 2 * y, 1 );
        m += row * row.transpose();
        rhs += row * ( x * x + y * y );
    }
    const Eigen::Vector3d sol = m.completeOrthogonalDecomposition().solve( rhs );
    const double r2 = sol[2] + sol[0] * sol[0] + sol[1] * sol[1];
    if ( !( r2 > 0 ) || !std::isfinite( r2 ) )
        return false;
    cx = sol[0] + mx;
    cy = sol[1] + my;
    rad = std::sqrt( r2 );
    return true;
}

Surf fitPlane( const std::vector<V3>& pts )
{
    Surf res;
    res.c = centroid( pts );
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    for ( const auto& p : pts )
    {
        const V3 d = p - res.c;
        const Eigen::Vector3d e( d.x, d.y, d.z );
        m += e * e.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es( m );
    const Eigen::Vector3d n = es.eigenvectors().col( 0 );
    res.a = V3( n[0], n[1], n[2] );
    return res;
}

std::optional<Surf> fitSphere( const std::vector<V3>& pts )
{
    const V3 mean = centroid( pts );
    Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
    Eigen::Vector4d rhs = Eigen::Vector4d::Zero();
    for ( const auto& p : pts )
    {
        const V3 d = p - mean;
        const Eigen::Vector4d row( 2 * d.x, 2 * d.y, 2 * d.z, 1 );
        m += row * row.transpose();
        rhs += row * d.lengthSq();
    }
    const Eigen::Vector4d sol = m.completeOrthogonalDecomposition().solve( rhs );
    const V3 c( sol[0], sol[1], sol[2] );
    const double r2 = sol[3] + c.lengthSq();
    if ( !( r2 > 0 ) || !std::isfinite( r2 ) )
        return {};
    Surf res;
    res.type = SurfType::Sphere;
    res.c = mean + c;
    res.r = std::sqrt( r2 );
    return res;
}

/// cylinder with the axis orthogonal to given normals
std::optional<Surf> fitCyl( const std::vector<V3>& pts, const std::vector<V3>& normals )
{
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    for ( const auto& n : normals )
    {
        const Eigen::Vector3d e( n.x, n.y, n.z );
        m += e * e.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es( m );
    const Eigen::Vector3d ax = es.eigenvectors().col( 0 );
    Surf res;
    res.type = SurfType::Cyl;
    res.a = V3( ax[0], ax[1], ax[2] ).normalized();
    V3 u, v;
    frame( res.a, u, v );
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

/// the line intersecting all normal lines: the axis of a surface of revolution
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

/// cone with given axis fitted to the points
std::optional<Surf> axisCone( const std::vector<V3>& pts, const V3& a, const V3& c, double* maxErr = nullptr )
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

/// torus with given axis fitted to the points
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

/// cone tangent to the planes passing through the points pc with given normals; unlike fitRevolution, it works for one strip of triangles
std::optional<Surf> fitConeByNormals( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals )
{
    // all normals make the same angle with the axis: dot( n, a ) = k
    Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
    Eigen::Matrix3d nn = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for ( size_t i = 0; i < normals.size(); ++i )
    {
        const auto& n = normals[i];
        const Eigen::Vector4d row( n.x, n.y, n.z, -1 );
        m += row * row.transpose();
        const Eigen::Vector3d e( n.x, n.y, n.z );
        nn += e * e.transpose();
        rhs += e * dot( n, pc[i] );
    }
    const Eigen::Vector4d v = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>( m ).eigenvectors().col( 0 );
    V3 a( v[0], v[1], v[2] );
    const double len = a.length();
    if ( !( len > 1e-9 ) )
        return {};
    a /= len;
    // the apex is on all tangent planes
    const Eigen::Vector3d ap = nn.completeOrthogonalDecomposition().solve( rhs );
    Surf res;
    res.type = SurfType::Cone;
    res.c = V3( ap[0], ap[1], ap[2] );
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

/// cone or torus fitted to the points, with the axis found from the normals in the points pc
std::optional<Surf> fitRevolution( const std::vector<V3>& pts, const std::vector<V3>& pc, const std::vector<V3>& normals, SurfType type )
{
    V3 a, c;
    if ( !revolutionAxis( pc, normals, a, c ) )
        return {};
    return type == SurfType::Cone ? axisCone( pts, a, c ) : axisTorus( pts, a, c );
}

/// minimizes the sum of squared distances from the points to the surface by Levenberg-Marquardt method
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

using Tri = std::array<int, 3>;

/// the number of boundary loops of a triangulation
int numBoundaryLoops( const std::vector<Tri>& tris )
{
    std::set<std::pair<int, int>> dirEdges;
    for ( const auto& t : tris )
        for ( int i = 0; i < 3; ++i )
            dirEdges.insert( { t[i], t[( i + 1 ) % 3] } );
    std::map<int, int> next;
    for ( const auto& [u, v] : dirEdges )
        if ( !dirEdges.count( { v, u } ) )
            next[u] = v;
    std::set<int> seen;
    int loops = 0;
    for ( const auto& [u, v] : next )
    {
        if ( seen.count( u ) )
            continue;
        ++loops;
        for ( int w = u; seen.insert( w ).second; )
        {
            auto it = next.find( w );
            if ( it == next.end() )
                break;
            w = it->second;
        }
    }
    return loops;
}

/// the part of the surface domain occupied by fitted points is [cMargin, 1-cMargin]^2, the rest is a natural extension
constexpr double cMargin = 0.06;

/// least squares conformal map with free boundary, rotated and scaled into [cMargin, 1-cMargin]^2 with u along the longest direction
std::vector<V2> lscm( const std::vector<V3>& pts, const std::vector<Tri>& tris )
{
    const int n = int( pts.size() );
    // pin two vertices most distant along the main direction
    const V3 mean = centroid( pts );
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for ( const auto& p : pts )
    {
        const V3 d = p - mean;
        const Eigen::Vector3d e( d.x, d.y, d.z );
        cov += e * e.transpose();
    }
    const Eigen::Vector3d ax = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>( cov ).eigenvectors().col( 2 );
    const V3 axis( ax[0], ax[1], ax[2] );
    int pa = 0, pb = 0;
    for ( int i = 0; i < n; ++i )
    {
        const double t = dot( pts[i], axis );
        if ( t < dot( pts[pa], axis ) )
            pa = i;
        if ( t > dot( pts[pb], axis ) )
            pb = i;
    }
    std::vector<int> col( 2 * n, -1 );
    int numFree = 0;
    for ( int i = 0; i < 2 * n; ++i )
        if ( i % n != pa && i % n != pb )
            col[i] = numFree++;

    std::vector<Eigen::Triplet<double>> trips;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero( 2 * tris.size() );
    int row = 0;
    for ( const auto& t : tris )
    {
        const V3 e1 = pts[t[1]] - pts[t[0]], e2 = pts[t[2]] - pts[t[0]];
        const double l1 = e1.length();
        const V3 nrm = cross( e1, e2 );
        const double ar = nrm.length();
        if ( !( l1 > 0 ) || !( ar > 0 ) )
            continue;
        const V3 ex = e1 / l1, ey = cross( nrm / ar, ex );
        const double x[3] = { 0, l1, dot( e2, ex ) }, y[3] = { 0, 0, dot( e2, ey ) };
        const double w = 1 / std::sqrt( ar );
        for ( int j = 0; j < 3; ++j )
        {
            const int a = ( j + 1 ) % 3, b = ( j + 2 ) % 3;
            const double wr = ( x[b] - x[a] ) * w, wi = ( y[b] - y[a] ) * w;
            const int v = t[j];
            // the pinned vertices have (u,v) = (0,0) and (1,0)
            auto add = [&]( int r, int unknown, double coeff )
            {
                if ( col[unknown] >= 0 )
                    trips.emplace_back( r, col[unknown], coeff );
                else if ( unknown == pb )
                    rhs[r] -= coeff;
            };
            add( row, v, wr );
            add( row, n + v, -wi );
            add( row + 1, v, wi );
            add( row + 1, n + v, wr );
        }
        row += 2;
    }
    Eigen::SparseMatrix<double> m( 2 * tris.size(), numFree );
    m.setFromTriplets( trips.begin(), trips.end() );
    const Eigen::SparseMatrix<double> mtm = m.transpose() * m;
    const Eigen::VectorXd sol = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>( mtm ).solve( m.transpose() * rhs );

    std::vector<V2> uv( n );
    for ( int i = 0; i < n; ++i )
    {
        uv[i].x = col[i] >= 0 ? sol[col[i]] : ( i == pb ? 1.0 : 0.0 );
        uv[i].y = col[n + i] >= 0 ? sol[col[n + i]] : 0.0;
    }
    V2 c;
    for ( const auto& p : uv )
        c += p / double( n );
    Eigen::Matrix2d cov2 = Eigen::Matrix2d::Zero();
    for ( auto& p : uv )
    {
        p -= c;
        const Eigen::Vector2d e( p.x, p.y );
        cov2 += e * e.transpose();
    }
    const Eigen::Matrix2d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>( cov2 ).eigenvectors();
    V2 lo( DBL_MAX, DBL_MAX ), hi( -DBL_MAX, -DBL_MAX );
    for ( auto& p : uv )
    {
        p = V2( p.x * ev( 0, 1 ) + p.y * ev( 1, 1 ), p.x * ev( 0, 0 ) + p.y * ev( 1, 0 ) );
        lo = V2( std::min( lo.x, p.x ), std::min( lo.y, p.y ) );
        hi = V2( std::max( hi.x, p.x ), std::max( hi.y, p.y ) );
    }
    for ( auto& p : uv )
    {
        p.x = cMargin + ( 1 - 2 * cMargin ) * ( p.x - lo.x ) / std::max( hi.x - lo.x, 1e-30 );
        p.y = cMargin + ( 1 - 2 * cMargin ) * ( p.y - lo.y ) / std::max( hi.y - lo.y, 1e-30 );
    }
    return uv;
}

/// weighted least squares fit of a B-spline surface with nu * nv control points to the points with known parameters
std::shared_ptr<BSplineSurf> lsqBSpline( const std::vector<V3>& pts, const std::vector<V2>& uv, const std::vector<double>& weights, int nu, int nv, double lambda )
{
    auto res = std::make_shared<BSplineSurf>();
    res->nu = nu;
    res->nv = nv;
    res->tu = clampedKnots( nu, BSplineSurf::deg );
    res->tv = clampedKnots( nv, BSplineSurf::deg );
    const int k = nu * nv;
    Eigen::MatrixXd ata = Eigen::MatrixXd::Zero( k, k );
    Eigen::MatrixXd atb = Eigen::MatrixXd::Zero( k, 3 );
    for ( size_t i = 0; i < pts.size(); ++i )
    {
        double bu[4], bv[4];
        const int iu = bsplineBasis( res->tu, BSplineSurf::deg, nu, uv[i].x, bu, nullptr );
        const int iv = bsplineBasis( res->tv, BSplineSurf::deg, nv, uv[i].y, bv, nullptr );
        int idx[16];
        double val[16];
        for ( int a = 0; a < 4; ++a )
            for ( int b = 0; b < 4; ++b )
            {
                idx[a * 4 + b] = ( iu + a ) * nv + iv + b;
                val[a * 4 + b] = bu[a] * bv[b] * weights[i];
            }
        for ( int a = 0; a < 16; ++a )
        {
            for ( int b = 0; b < 16; ++b )
                ata( idx[a], idx[b] ) += val[a] * val[b];
            for ( int d = 0; d < 3; ++d )
                atb( idx[a], d ) += val[a] * weights[i] * pts[i][d];
        }
    }
    // smoothing: second differences of the control net
    const int numRows = ( nu - 2 ) * nv + nu * ( nv - 2 );
    const double w2 = lambda * double( pts.size() ) / std::max( numRows, 1 );
    auto addRow = [&]( int i0, int i1, int i2 )
    {
        const int idx[3] = { i0, i1, i2 };
        const double val[3] = { 1, -2, 1 };
        for ( int a = 0; a < 3; ++a )
            for ( int b = 0; b < 3; ++b )
                ata( idx[a], idx[b] ) += w2 * val[a] * val[b];
    };
    for ( int a = 0; a < nu; ++a )
        for ( int b = 0; b < nv; ++b )
        {
            if ( a > 0 && a + 1 < nu )
                addRow( ( a - 1 ) * nv + b, a * nv + b, ( a + 1 ) * nv + b );
            if ( b > 0 && b + 1 < nv )
                addRow( a * nv + b - 1, a * nv + b, a * nv + b + 1 );
        }
    const Eigen::MatrixXd sol = ata.ldlt().solve( atb );
    res->cp.resize( k );
    for ( int i = 0; i < k; ++i )
        res->cp[i] = V3( sol( i, 0 ), sol( i, 1 ), sol( i, 2 ) );
    return res;
}

/// approximates a triangulation with disk topology by a B-spline surface passing near its vertices;
/// \return nullptr if the triangulation is not a disk or has too few vertices
std::shared_ptr<BSplineSurf> fitBSpline( const std::vector<V3>& pts, const std::vector<Tri>& tris, double tol )
{
    if ( numBoundaryLoops( tris ) != 1 )
        return {};
    const auto uv0 = lscm( pts, tris );
    for ( const auto& p : uv0 )
        if ( !std::isfinite( p.x + p.y ) )
            return {};

    // points inside the triangles much larger than the typical one: without them the surface is free to bulge there
    std::vector<double> lens;
    lens.reserve( tris.size() * 3 );
    for ( const auto& t : tris )
        for ( int i = 0; i < 3; ++i )
            lens.push_back( ( pts[t[i]] - pts[t[( i + 1 ) % 3]] ).length() );
    auto mid = lens.begin() + lens.size() / 2;
    std::nth_element( lens.begin(), mid, lens.end() );
    const double med = *mid;
    auto allPts = pts;
    auto allUV = uv0;
    for ( const auto& t : tris )
    {
        double longest = 0;
        for ( int i = 0; i < 3; ++i )
            longest = std::max( longest, ( pts[t[i]] - pts[t[( i + 1 ) % 3]] ).length() );
        if ( longest <= 3 * med )
            continue;
        const int n = (int)std::min( 8.0, std::ceil( longest / ( 1.5 * med ) ) );
        for ( int i = 0; i <= n; ++i )
            for ( int j = 0; i + j <= n; ++j )
            {
                const int k = n - i - j;
                if ( std::max( { i, j, k } ) == n )
                    continue;
                const double b0 = double( i ) / n, b1 = double( j ) / n, b2 = double( k ) / n;
                allPts.push_back( b0 * pts[t[0]] + b1 * pts[t[1]] + b2 * pts[t[2]] );
                allUV.push_back( b0 * uv0[t[0]] + b1 * uv0[t[1]] + b2 * uv0[t[2]] );
            }
    }
    std::vector<double> weights( allPts.size(), 0.1 );
    std::fill( weights.begin(), weights.begin() + pts.size(), 1.0 );

    std::shared_ptr<BSplineSurf> best;
    static constexpr int sizes[][2] = { { 6, 4 }, { 8, 4 }, { 10, 5 }, { 14, 5 }, { 18, 6 }, { 24, 6 }, { 32, 7 }, { 40, 8 } };
    for ( const auto& sz : sizes )
    {
        if ( sz[0] * sz[1] > 0.7 * pts.size() )
            break;
        auto uv = allUV;
        std::shared_ptr<BSplineSurf> s;
        for ( int it = 0; it < 8; ++it )
        {
            s = lsqBSpline( allPts, uv, weights, sz[0], sz[1], 1e-7 );
            for ( size_t i = 0; i < allPts.size(); ++i )
                uv[i] = s->project( allPts[i], &uv[i], 4 );
        }
        s->buildGrid();
        // projection from scratch also catches folds of the surface
        for ( const auto& p : pts )
            s->err = std::max( s->err, s->absDist( p ) );
        if ( !std::isfinite( s->err ) )
            continue;
        if ( !best || s->err < best->err )
            best = s;
        if ( s->err < tol )
            break;
    }
    return best;
}

/// a part of the mesh lying on one surface
struct Region
{
    Surf surf;
    int seg = 0;      ///< the segment of the mesh given by the user
    int numFaces = 0;
    double area = 0;
};

/// the mesh in the form convenient for the algorithms below, and its subdivision on regions
struct Data
{
    int nf = 0;
    std::vector<V3> pts;       ///< by vertex
    std::vector<Tri> tris;     ///< by face; {-1,-1,-1} for invalid faces
    std::vector<Tri> nbr;      ///< faces across the edges (v0v1, v1v2, v2v0), -1 if no face there
    std::vector<V3> normals, centers;
    std::vector<double> areas;
    std::vector<int> seg;      ///< user segment by face, -1 for invalid faces
    std::vector<int> sub;      ///< region by face
    std::vector<Region> regions;
    double tol = 0, total = 0, diag = 0;
    int numSegs = 0;

    bool valid( int f ) const { return tris[f][0] >= 0; }

    /// the neighbour across i-th edge if it is in the same segment
    int nbrSameSeg( int f, int i ) const
    {
        const int g = nbr[f][i];
        return g >= 0 && seg[g] == seg[f] ? g : -1;
    }

    void updateStats()
    {
        for ( auto& r : regions )
        {
            r.numFaces = 0;
            r.area = 0;
        }
        for ( int f = 0; f < nf; ++f )
            if ( valid( f ) )
            {
                ++regions[sub[f]].numFaces;
                regions[sub[f]].area += areas[f];
            }
    }

    std::vector<std::vector<int>> facesOfRegions() const
    {
        std::vector<std::vector<int>> res( regions.size() );
        for ( int f = 0; f < nf; ++f )
            if ( valid( f ) )
                res[sub[f]].push_back( f );
        return res;
    }

    /// all triangles' vertices of given faces, with repetitions
    std::vector<V3> triPoints( const std::vector<int>& faces ) const
    {
        std::vector<V3> res;
        res.reserve( faces.size() * 3 );
        for ( int f : faces )
            for ( int v : tris[f] )
                res.push_back( pts[v] );
        return res;
    }

    /// distinct vertices of given faces
    std::vector<int> vertsOf( const std::vector<int>& faces ) const
    {
        std::vector<int> res;
        res.reserve( faces.size() * 3 );
        for ( int f : faces )
            for ( int v : tris[f] )
                res.push_back( v );
        std::sort( res.begin(), res.end() );
        res.erase( std::unique( res.begin(), res.end() ), res.end() );
        return res;
    }

    /// positions of distinct vertices of given faces, evenly subsampled if there are more than maxNum of them
    std::vector<V3> pointsOf( const std::vector<int>& faces, size_t maxNum = SIZE_MAX ) const
    {
        const auto vs = vertsOf( faces );
        std::vector<V3> res;
        if ( vs.size() <= maxNum )
        {
            res.reserve( vs.size() );
            for ( int v : vs )
                res.push_back( pts[v] );
        }
        else
        {
            res.reserve( maxNum );
            for ( size_t i = 0; i < maxNum; ++i )
                res.push_back( pts[vs[i * ( vs.size() - 1 ) / ( maxNum - 1 )]] );
        }
        return res;
    }

    /// area-weighted average distance from the centers of given faces to the surface
    double centersDeviation( const Surf& s, const std::vector<int>& faces ) const
    {
        double sum = 0, sumA = 0;
        for ( int f : faces )
        {
            sum += std::abs( s.dist( centers[f] ) ) * areas[f];
            sumA += areas[f];
        }
        return sumA > 0 ? sum / sumA : 0;
    }

    /// facet normals must not deviate much from the normals of the surface
    bool normalsAgree( const Surf& s, int f, bool checkVerts ) const
    {
        if ( std::abs( dot( s.normal( centers[f] ), normals[f] ) ) <= cCos40 )
            return false;
        if ( checkVerts && s.type != SurfType::Plane && s.type != SurfType::BSpline )
        {
            // the middles of facet's edges must not be too far from the surface relative to its radius,
            // otherwise it is rather a prism or a pyramid than a coarse cylinder or cone
            for ( int i = 0; i < 3; ++i )
            {
                const V3 m = ( pts[tris[f][i]] + pts[tris[f][( i + 1 ) % 3]] ) / 2.0;
                double rad = s.r;
                if ( s.type == SurfType::Cone )
                {
                    const V3 dm = m - s.c;
                    rad = ( dm - s.a * dot( dm, s.a ) ).length();
                }
                else if ( s.type == SurfType::Torus )
                    rad = s.R + s.r;
                if ( std::abs( s.dist( m ) ) > 0.1 * rad )
                    return false;
            }
        }
        if ( checkVerts && s.type != SurfType::Plane && s.type != SurfType::BSpline )
            for ( int v : tris[f] )
            {
                if ( s.type == SurfType::Cone && ( pts[v] - s.c ).length() < 5 * tol )
                    continue; // no normal in the apex
                // the limit is below 30 degrees to distinguish a hexagonal prism from a cylinder
                if ( std::abs( dot( s.normal( pts[v] ), normals[f] ) ) < cCosVert )
                    return false;
            }
        return true;
    }

    bool faceOnSurf( const Surf& s, int f, double maxDist ) const
    {
        for ( int v : tris[f] )
            if ( !( std::abs( s.dist( pts[v] ) ) < maxDist ) )
                return false;
        return normalsAgree( s, f, true );
    }
};

Data makeData( const Mesh& mesh, const MeshToStepSettings& settings )
{
    Data d;
    const auto& topology = mesh.topology;
    d.nf = int( topology.lastValidFace() ) + 1;
    d.pts.resize( mesh.points.size() );
    for ( int v = 0; v < (int)d.pts.size(); ++v )
        d.pts[v] = V3( mesh.points[VertId( v )] );
    d.tris.assign( d.nf, Tri{ -1, -1, -1 } );
    d.nbr.assign( d.nf, Tri{ -1, -1, -1 } );
    d.normals.resize( d.nf );
    d.centers.resize( d.nf );
    d.areas.resize( d.nf, 0.0 );
    std::vector<std::array<bool, 3>> cut( d.nf, { false, false, false } );
    Box3d box;
    for ( FaceId f : topology.getValidFaces() )
    {
        EdgeId e = topology.edgeWithLeft( f );
        for ( int i = 0; i < 3; ++i )
        {
            d.tris[f][i] = topology.org( e );
            d.nbr[f][i] = topology.right( e ).valid() ? int( topology.right( e ) ) : -1;
            cut[f][i] = settings.faceBoundaries && settings.faceBoundaries->test( e.undirected() );
            e = topology.prev( e.sym() );
        }
        const V3& a = d.pts[d.tris[f][0]];
        const V3& b = d.pts[d.tris[f][1]];
        const V3& c = d.pts[d.tris[f][2]];
        const V3 n = cross( b - a, c - a );
        const double len = n.length();
        d.areas[f] = len / 2;
        d.normals[f] = len > 0 ? n / len : V3();
        d.centers[f] = ( a + b + c ) / 3.0;
        d.total += d.areas[f];
        box.include( a );
        box.include( b );
        box.include( c );
    }
    d.diag = box.valid() ? box.diagonal() : 0;
    d.tol = settings.tolerance > 0 ? settings.tolerance : 2.5e-6 * d.diag;

    // segments are connected components of faces not separated by the given edges
    d.seg.assign( d.nf, -1 );
    std::vector<int> stack;
    for ( int f0 = 0; f0 < d.nf; ++f0 )
    {
        if ( !d.valid( f0 ) || d.seg[f0] >= 0 )
            continue;
        d.seg[f0] = d.numSegs;
        stack.push_back( f0 );
        while ( !stack.empty() )
        {
            const int f = stack.back();
            stack.pop_back();
            for ( int i = 0; i < 3; ++i )
            {
                const int g = d.nbr[f][i];
                if ( g >= 0 && !cut[f][i] && d.seg[g] < 0 )
                {
                    d.seg[g] = d.numSegs;
                    stack.push_back( g );
                }
            }
        }
        ++d.numSegs;
    }
    d.sub.assign( d.nf, -1 );
    return d;
}

/// \return true if two surfaces of the same analytic type coincide within the tolerance
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

struct Axis
{
    V3 dir, pos;
};

/// \return true if the axis of the surface coincides with one of given axes
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

/// finds regions of the mesh lying on planes, cylinders, cones, spheres and tori by growing them from seed triangles
class Grower
{
public:
    Grower( Data& d, const ProgressCallback& cb ) : d_( d ), cb_( cb )
    {
        avail_.resize( d.nf, 0 );
        fail_.resize( d.nf, 0 );
        faceStamp_.resize( d.nf, 0 );
        coveredStamp_.resize( d.nf, 0 );
    }

    /// \return false if the operation was canceled
    bool run();

private:
    struct Found
    {
        double area = 0, dev = 0;
        Surf surf;
        std::vector<int> faces;
    };

    static constexpr size_t cMinVerts = 12;
    static constexpr int cMinStrongFaces = 20;

    std::optional<std::vector<int>> growOnce_( Surf& s, int seed, std::initializer_list<double> sched );
    std::optional<std::vector<int>> grow_( Surf& s, int seed );
    std::vector<int> patch_( int seed, int rings ) const;
    std::vector<int> sample_( const std::vector<int>& from, size_t num );
    std::vector<V3> normalsOf_( const std::vector<int>& faces ) const;
    std::vector<V3> centersOf_( const std::vector<int>& faces ) const;
    void addIfFits_( std::vector<Surf>& out, std::optional<Surf> s, const std::vector<V3>& pts, bool allowRefine ) const;
    void candidates_( int seed, std::vector<Surf>& out );
    std::optional<Surf> simpler_( const Surf& s, const std::vector<int>& faces ) const;
    bool better_( const Found& x, const Found& y ) const;
    bool strong_( const Region& r ) const;
    void processSegment_( int seg, const std::vector<int>& segFaces );

    Data& d_;
    ProgressCallback cb_;
    std::vector<char> avail_;
    std::vector<int> fail_;
    std::vector<int> faceStamp_, coveredStamp_;
    int stamp_ = 0;
    std::mt19937 rng_{ 1 };
    std::vector<Surf> known_; ///< curved surfaces found in the previous pass, they may continue into other segments
    std::vector<Axis> lib_;   ///< axes of cylinders and cones found in the previous pass
    int pass_ = 1;
};

std::optional<std::vector<int>> Grower::growOnce_( Surf& s, int seed, std::initializer_list<double> sched )
{
    std::vector<int> reg;
    for ( int it = 0; it < 10; ++it )
    {
        const double tol = d_.tol * ( it < (int)sched.size() ? sched.begin()[it] : 1.0 );
        if ( !d_.faceOnSurf( s, seed, tol ) )
            return {};
        ++stamp_;
        std::vector<int> cur{ seed };
        faceStamp_[seed] = stamp_;
        for ( size_t i = 0; i < cur.size(); ++i )
            for ( int k = 0; k < 3; ++k )
            {
                const int g = d_.nbrSameSeg( cur[i], k );
                if ( g < 0 || !avail_[g] || faceStamp_[g] == stamp_ )
                    continue;
                // a curved surface is not expected to be tessellated with large angles between adjacent facets:
                // it is rather a prism or a pyramid
                if ( s.type != SurfType::Plane && dot( d_.normals[cur[i]], d_.normals[g] ) < cCosFacets )
                    continue;
                faceStamp_[g] = stamp_;
                if ( d_.faceOnSurf( s, g, tol ) )
                    cur.push_back( g );
            }
        std::sort( cur.begin(), cur.end() );
        if ( it >= (int)sched.size() && cur == reg )
            break;
        reg = std::move( cur );
        if ( s.known )
            break;
        const auto numVerts = d_.vertsOf( reg ).size();
        if ( s.type != SurfType::Plane && numVerts < cMinVerts && it == 0 )
            return {};
        if ( numVerts < 4 )
            break;
        s = refine( s, d_.pointsOf( reg, 600 ) );
    }
    return reg;
}

std::optional<std::vector<int>> Grower::grow_( Surf& s, int seed )
{
    const Surf s0 = s;
    auto res = growOnce_( s, seed, {} );
    if ( !res && s0.type != SurfType::Plane && !s0.known )
    {
        // coarse facets give a poor start: begin with a large tolerance and decrease it gradually
        s = s0;
        res = growOnce_( s, seed, { 50, 10, 3 } );
    }
    return res;
}

std::vector<int> Grower::patch_( int seed, int rings ) const
{
    std::vector<int> res{ seed };
    size_t begin = 0;
    for ( int r = 0; r < rings; ++r )
    {
        const size_t end = res.size();
        for ( size_t i = begin; i < end; ++i )
            for ( int k = 0; k < 3; ++k )
            {
                const int g = d_.nbrSameSeg( res[i], k );
                if ( g >= 0 && avail_[g] && std::find( res.begin(), res.end(), g ) == res.end() )
                    res.push_back( g );
            }
        begin = end;
    }
    return res;
}

std::vector<int> Grower::sample_( const std::vector<int>& from, size_t num )
{
    auto v = from;
    num = std::min( num, v.size() );
    for ( size_t i = 0; i < num; ++i )
        std::swap( v[i], v[i + rng_() % ( v.size() - i )] );
    v.resize( num );
    return v;
}

std::vector<V3> Grower::normalsOf_( const std::vector<int>& faces ) const
{
    std::vector<V3> res;
    res.reserve( faces.size() );
    for ( int f : faces )
        res.push_back( d_.normals[f] );
    return res;
}

std::vector<V3> Grower::centersOf_( const std::vector<int>& faces ) const
{
    std::vector<V3> res;
    res.reserve( faces.size() );
    for ( int f : faces )
        res.push_back( d_.centers[f] );
    return res;
}

void Grower::addIfFits_( std::vector<Surf>& out, std::optional<Surf> s, const std::vector<V3>& pts, bool allowRefine ) const
{
    if ( !s || !s->finite() )
        return;
    double e = s->maxDist( pts );
    if ( allowRefine && e > d_.tol && e <= 100 * d_.tol )
    {
        s = refine( *s, pts );
        e = s->maxDist( pts );
    }
    if ( s->finite() && e <= d_.tol )
        out.push_back( *s );
}

void Grower::candidates_( int seed, std::vector<Surf>& out )
{
    const auto& d = d_;
    Surf plane;
    plane.c = d.centers[seed];
    plane.a = d.normals[seed];
    out.push_back( plane );
    for ( int rings : { 1, 2 } )
    {
        const auto pts = d.pointsOf( patch_( seed, rings ) );
        if ( pts.size() >= 4 )
            if ( auto s = fitPlane( pts ); s.maxDist( pts ) < d.tol )
                out.push_back( s );
    }

    // a cylinder through two triangles, a cone through three triangles
    auto others = patch_( seed, 3 );
    others.erase( others.begin() );
    if ( others.size() >= 2 )
    {
        for ( int q : sample_( others, 8 ) )
        {
            V3 a = cross( d.normals[seed], d.normals[q] );
            const double len = a.length();
            if ( len < 0.02 )
                continue;
            a /= len;
            V3 u, v;
            frame( a, u, v );
            std::vector<double> xs, ys;
            for ( int f : { seed, q } )
                for ( int vert : d.tris[f] )
                {
                    xs.push_back( dot( d.pts[vert], u ) );
                    ys.push_back( dot( d.pts[vert], v ) );
                }
            Surf s;
            s.type = SurfType::Cyl;
            s.a = a;
            double cx, cy;
            if ( fitCircle2( xs, ys, cx, cy, s.r ) )
            {
                s.c = cx * u + cy * v;
                out.push_back( s );
            }
        }
        for ( int i = 0; i < 8; ++i )
        {
            const auto q = sample_( others, 2 );
            const int fs[3] = { seed, q[0], q[1] };
            const V3 n0 = d.normals[fs[0]], n1 = d.normals[fs[1]], n2 = d.normals[fs[2]];
            V3 a = cross( n0 - n1, n0 - n2 );
            const double len = a.length();
            if ( len < 1e-3 )
                continue;
            a /= len;
            Eigen::Matrix3d m;
            m << n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n2.x, n2.y, n2.z;
            if ( std::abs( dot( n0, a ) ) < 0.02 || std::abs( m.determinant() ) < 1e-6 )
                continue;
            const Eigen::Vector3d ap = m.inverse() * Eigen::Vector3d( dot( n0, d.centers[fs[0]] ), dot( n1, d.centers[fs[1]] ), dot( n2, d.centers[fs[2]] ) );
            const V3 apex( ap[0], ap[1], ap[2] );
            std::vector<double> hs;
            double sumTan = 0, minH = DBL_MAX;
            for ( int f : fs )
                for ( int vert : d.tris[f] )
                    hs.push_back( dot( d.pts[vert] - apex, a ) );
            auto sorted = hs;
            std::sort( sorted.begin(), sorted.end() );
            const double sign = sorted[sorted.size() / 2] < 0 ? -1 : 1;
            a *= sign;
            for ( int f : fs )
                for ( int vert : d.tris[f] )
                {
                    const V3 dv = d.pts[vert] - apex;
                    const double h = dot( dv, a );
                    minH = std::min( minH, h );
                    sumTan += ( dv - h * a ).length() / h;
                }
            if ( minH <= 0 )
                continue;
            Surf s;
            s.type = SurfType::Cone;
            s.c = apex;
            s.a = a;
            s.r = std::atan( sumTan / 9 );
            if ( s.finite() )
                out.push_back( s );
        }
    }

    // coarse facets: a cylinder through three triangles of similar size (two triangles give a poor radius)
    {
        std::vector<int> big;
        for ( int f : patch_( seed, 6 ) )
            if ( f != seed && d.areas[f] >= 0.2 * d.areas[seed] )
                big.push_back( f );
        if ( big.size() >= 2 )
        {
            if ( big.size() > 12 )
            {
                std::sort( big.begin(), big.end(), [&]( int x, int y )
                {
                    return ( d.centers[x] - d.centers[seed] ).lengthSq() < ( d.centers[y] - d.centers[seed] ).lengthSq();
                } );
                big.resize( 12 );
            }
            const size_t numTries = std::min<size_t>( 12, big.size() * ( big.size() - 1 ) / 2 );
            for ( size_t i = 0; i < numTries; ++i )
            {
                const auto q = sample_( big, 2 );
                const std::vector<int> fs{ seed, q[0], q[1] };
                const auto pts = d.pointsOf( fs );
                if ( auto s0 = fitCyl( pts, normalsOf_( fs ) ) )
                {
                    const Surf s = refine( *s0, pts );
                    if ( s.finite() && s.r < 100 * d.diag && s.maxDist( pts ) < d.tol )
                        out.push_back( s );
                }
            }
        }
    }

    // surfaces fitted to all triangles around the seed
    for ( int rings : { 2, 4, 8 } )
    {
        const auto p = patch_( seed, rings );
        if ( p.size() < 6 )
            continue;
        const auto pts = d.pointsOf( p );
        if ( pts.size() < cMinVerts )
            continue;
        const auto ns = normalsOf_( p );
        const auto cs = centersOf_( p );
        addIfFits_( out, fitCyl( pts, ns ), pts, true );
        addIfFits_( out, fitRevolution( pts, cs, ns, SurfType::Cone ), pts, true );
        addIfFits_( out, fitConeByNormals( pts, cs, ns ), pts, true );
        addIfFits_( out, fitSphere( pts ), pts, true );
        addIfFits_( out, fitRevolution( pts, cs, ns, SurfType::Torus ), pts, true );
    }

    // tori and cones around the known axes
    if ( !lib_.empty() )
        for ( int rings : { 1, 2, 4 } )
        {
            const auto pts = d.pointsOf( patch_( seed, rings ) );
            if ( pts.size() < 6 )
                continue;
            for ( const auto& ax : lib_ )
            {
                if ( auto s = axisTorus( pts, ax.dir, ax.pos ); s && s->maxDist( pts ) < d.tol )
                    out.push_back( *s );
                double err = 0;
                if ( auto s = axisCone( pts, ax.dir, ax.pos, &err ); s && err < d.tol && std::tan( s->r ) > 1e-4 && std::tan( s->r ) < 1e4 )
                    out.push_back( *s );
            }
        }

    for ( const auto& s : known_ )
    {
        bool fits = true;
        for ( int vert : d.tris[seed] )
            fits = fits && std::abs( s.dist( d.pts[vert] ) ) < d.tol;
        if ( fits )
            out.push_back( s );
    }
}

/// a simpler surface passing through all vertices of the region, if any
std::optional<Surf> Grower::simpler_( const Surf& s, const std::vector<int>& faces ) const
{
    const auto pts = d_.pointsOf( faces, 600 );
    const auto ns = normalsOf_( faces );
    for ( int variant = 0; variant < 4; ++variant )
    {
        const auto type = variant == 0 ? SurfType::Plane : variant == 1 ? SurfType::Cyl : SurfType::Cone;
        if ( type >= s.type )
            break;
        std::optional<Surf> q;
        if ( variant == 0 )
            q = fitPlane( pts );
        else if ( variant == 1 )
            q = fitCyl( pts, ns );
        else if ( variant == 2 )
            q = fitConeByNormals( pts, centersOf_( faces ), ns );
        else
            q = fitRevolution( pts, centersOf_( faces ), ns, SurfType::Cone );
        if ( !q )
            continue;
        if ( type != SurfType::Plane )
            q = refine( *q, pts );
        if ( !q->finite() || !( q->maxDist( pts ) < d_.tol ) )
            continue;
        bool ok = true;
        for ( int f : faces )
            ok = ok && std::abs( dot( q->normal( d_.centers[f] ), d_.normals[f] ) ) > cCos40;
        if ( ok )
            return q;
    }
    return {};
}

/// for equal areas: the simpler type wins unless it leaves the facets much farther from the surface
bool Grower::better_( const Found& x, const Found& y ) const
{
    if ( x.surf.type == y.surf.type )
        return x.dev < y.dev;
    const bool xSimpler = x.surf.type < y.surf.type;
    const Found& lo = xSimpler ? x : y;
    const Found& hi = xSimpler ? y : x;
    const bool loWins = lo.dev <= 3 * hi.dev + 0.5 * d_.tol;
    return loWins == xSimpler;
}

bool Grower::strong_( const Region& r ) const
{
    if ( r.surf.type == SurfType::Plane )
        return r.numFaces >= 3;
    if ( r.numFaces < cMinStrongFaces )
        return false;
    return r.surf.type != SurfType::Torus || isCoaxial( r.surf, lib_, d_.tol );
}

void Grower::processSegment_( int seg, const std::vector<int>& segFaces )
{
    auto& d = d_;
    std::vector<int> faces;
    for ( int f : segFaces )
        if ( d.sub[f] < 0 )
        {
            faces.push_back( f );
            avail_[f] = 1;
        }
    auto addRegion = [&]( const Surf& s, const std::vector<int>& fs )
    {
        Region r;
        r.surf = s;
        r.seg = seg;
        for ( int f : fs )
        {
            d.sub[f] = (int)d.regions.size();
            avail_[f] = 0;
            ++r.numFaces;
            r.area += d.areas[f];
        }
        d.regions.push_back( std::move( r ) );
    };
    std::vector<Surf> cands;
    while ( !faces.empty() )
    {
        // first the largest triangles not failed before
        std::sort( faces.begin(), faces.end(), [&]( int x, int y )
        {
            return std::tuple( fail_[x], -d.areas[x], x ) < std::tuple( fail_[y], -d.areas[y], y );
        } );
        std::vector<Found> found;
        bool anyTried = false;
        for ( int phase = 0; phase < 4; ++phase )
        {
            // phase 0: surfaces of all types from the seeds not failed before;
            // next phases, after all seeds have failed: only planes of at least minPlane triangles
            const bool full = phase == 0;
            const int minPlane = full ? 3 : 4 - phase;
            if ( !full && anyTried )
                break;
            const int coveredStamp = ++stamp_;
            int tried = 0;
            for ( int seed : faces )
            {
                if ( coveredStamp_[seed] == coveredStamp )
                    continue;
                if ( d.areas[seed] <= 0 )
                    continue;
                if ( full )
                {
                    if ( fail_[seed] > 0 )
                        break;
                    ++tried;
                    if ( tried > ( pass_ == 1 ? 300 : 60 ) || ( !found.empty() && tried > 40 ) )
                        break;
                    anyTried = true;
                }
                cands.clear();
                if ( full )
                    candidates_( seed, cands );
                else
                {
                    Surf plane;
                    plane.c = d.centers[seed];
                    plane.a = d.normals[seed];
                    cands.push_back( plane );
                }
                std::optional<Found> best;
                for ( auto& s : cands )
                {
                    auto reg = grow_( s, seed );
                    if ( !reg )
                        continue;
                    if ( s.type == SurfType::Plane )
                    {
                        if ( (int)reg->size() < minPlane )
                            continue;
                    }
                    else if ( !s.known && d.vertsOf( *reg ).size() < cMinVerts )
                        continue;
                    Found fnd;
                    for ( int f : *reg )
                        fnd.area += d.areas[f];
                    const double q = d.tol / 20;
                    fnd.dev = std::round( d.centersDeviation( s, *reg ) / q ) * q;
                    fnd.surf = s;
                    fnd.faces = std::move( *reg );
                    if ( !best || fnd.area > best->area * 1.0001 || ( fnd.area > best->area * 0.9999 && better_( fnd, *best ) ) )
                        best = std::move( fnd );
                }
                if ( best && best->surf.type != SurfType::Plane && !best->surf.known )
                {
                    if ( auto q = simpler_( best->surf, best->faces ) )
                    {
                        auto reg = grow_( *q, seed );
                        if ( reg && reg->size() >= best->faces.size() )
                        {
                            best->area = 0;
                            for ( int f : *reg )
                                best->area += d.areas[f];
                            best->dev = 0;
                            best->surf = *q;
                            best->faces = std::move( *reg );
                        }
                    }
                }
                if ( best )
                {
                    for ( int f : best->faces )
                        coveredStamp_[f] = coveredStamp;
                    found.push_back( std::move( *best ) );
                }
                else if ( full )
                {
                    // do not start from the neighbours of the failed seed: most probably they are on the same freeform surface
                    for ( int f : patch_( seed, 3 ) )
                        ++fail_[f];
                }
            }
            if ( !found.empty() )
                break;
        }
        if ( found.empty() && anyTried )
            continue;
        if ( found.empty() )
        {
            // only degenerate triangles remain
            for ( int f : faces )
                addRegion( fitPlane( d.pointsOf( { f } ) ), { f } );
            break;
        }
        std::stable_sort( found.begin(), found.end(), []( const Found& x, const Found& y ) { return x.area > y.area; } );
        for ( const auto& fnd : found )
            if ( std::all_of( fnd.faces.begin(), fnd.faces.end(), [&]( int f ) { return avail_[f]; } ) )
                addRegion( fnd.surf, fnd.faces );
        std::erase_if( faces, [&]( int f ) { return !avail_[f]; } );
    }
}

bool Grower::run()
{
    MR_TIMER;
    auto& d = d_;
    std::vector<std::vector<int>> segFaces( d.numSegs );
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) )
            segFaces[d.seg[f]].push_back( f );
    for ( pass_ = 1; pass_ <= 2; ++pass_ )
    {
        if ( pass_ == 2 )
        {
            // keep well supported regions, redo the rest with the knowledge of found surfaces
            std::vector<int> map( d.regions.size(), -1 );
            std::vector<Region> kept;
            for ( size_t i = 0; i < d.regions.size(); ++i )
                if ( strong_( d.regions[i] ) )
                {
                    map[i] = (int)kept.size();
                    kept.push_back( d.regions[i] );
                }
            for ( int f = 0; f < d.nf; ++f )
                if ( d.valid( f ) )
                    d.sub[f] = map[d.sub[f]];
            d.regions = std::move( kept );
            std::fill( fail_.begin(), fail_.end(), 0 );
        }
        size_t doneFaces = 0;
        size_t numFaces = 0;
        for ( const auto& fs : segFaces )
            numFaces += fs.size();
        for ( int s = 0; s < d.numSegs; ++s )
        {
            processSegment_( s, segFaces[s] );
            doneFaces += segFaces[s].size();
            if ( !reportProgress( cb_, 0.5f * ( float( pass_ - 1 ) + float( doneFaces ) / float( numFaces ) ) ) )
                return false;
        }
        if ( pass_ == 1 )
        {
            for ( const auto& r : d.regions )
            {
                if ( r.surf.type == SurfType::Plane || r.numFaces < cMinStrongFaces )
                    continue;
                if ( std::none_of( known_.begin(), known_.end(), [&]( const Surf& k ) { return sameSurface( k, r.surf, 0.5 * d.tol ); } ) )
                {
                    known_.push_back( r.surf );
                    known_.back().known = true;
                }
                if ( r.surf.type == SurfType::Cyl || r.surf.type == SurfType::Cone )
                    addAxis( lib_, r.surf, d.tol );
            }
        }
    }
    for ( auto& r : d.regions )
        r.surf.known = false;
    return true;
}

/// improves the regions found by Grower, and replaces groups of unreliable regions with B-spline surfaces
class Cleaner
{
public:
    Cleaner( Data& d ) : d_( d ) {}
    void run();

private:
    using Pair = std::pair<int, int>;
    struct Border
    {
        double len = 0, angLen = 0;
        double angle() const { return len > 0 ? angLen / len : 0; }
    };

    bool strong_( const Region& r ) const;
    std::map<Pair, Border> borders_() const;
    void moveFaces_( int from, int to );
    void extendStrong_();
    std::optional<Surf> unionFit_( const std::vector<int>& fa, const std::vector<int>& fb, const Region& ra, const Region& rb ) const;
    void uniteUnderOneSurface_();
    void absorbLyingOnNeighbour_();
    void mergeSameSurface_();
    void fitFreeform_();
    void mergeZeroArea_();
    void splitOnComponents_();

    Data& d_;
    std::vector<Axis> axes_;
};

bool Cleaner::strong_( const Region& r ) const
{
    if ( r.numFaces == 0 )
        return false;
    if ( r.surf.type == SurfType::Plane )
        return r.numFaces >= 3;
    if ( r.surf.type == SurfType::BSpline )
        return true;
    if ( r.numFaces < 20 && r.area < 2e-4 * d_.total )
        return false;
    if ( r.surf.type != SurfType::Torus )
        return true;
    return isCoaxial( r.surf, axes_, d_.tol ) || ( r.surf.r <= 0.5 * r.surf.R && r.numFaces >= 100 );
}

/// common borders of all pairs of adjacent regions from the same segment
std::map<Cleaner::Pair, Cleaner::Border> Cleaner::borders_() const
{
    const auto& d = d_;
    std::map<Pair, Border> res;
    for ( int f = 0; f < d.nf; ++f )
    {
        if ( !d.valid( f ) )
            continue;
        for ( int i = 0; i < 3; ++i )
        {
            const int g = d.nbrSameSeg( f, i );
            if ( g < f || d.sub[f] == d.sub[g] )
                continue;
            auto& b = res[std::minmax( d.sub[f], d.sub[g] )];
            const double len = ( d.pts[d.tris[f][i]] - d.pts[d.tris[f][( i + 1 ) % 3]] ).length();
            b.len += len;
            b.angLen += len * std::acos( std::clamp( dot( d.normals[f], d.normals[g] ), -1.0, 1.0 ) ) * 180 / PI;
        }
    }
    return res;
}

void Cleaner::moveFaces_( int from, int to )
{
    for ( int f = 0; f < d_.nf; ++f )
        if ( d_.valid( f ) && d_.sub[f] == from )
            d_.sub[f] = to;
}

/// strong regions take over adjacent weak triangles that lie on their surface (looser distance, strict normal)
void Cleaner::extendStrong_()
{
    auto& d = d_;
    std::vector<char> isStrong( d.regions.size() );
    for ( size_t i = 0; i < d.regions.size(); ++i )
        isStrong[i] = strong_( d.regions[i] );
    std::vector<int> queue;
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) && isStrong[d.sub[f]] )
            queue.push_back( f );
    while ( !queue.empty() )
    {
        const int f = queue.back();
        queue.pop_back();
        const int r = d.sub[f];
        const Surf& s = d.regions[r].surf;
        for ( int i = 0; i < 3; ++i )
        {
            const int g = d.nbrSameSeg( f, i );
            if ( g < 0 || d.sub[g] == r || isStrong[d.sub[g]] )
                continue;
            double maxDist = 0, maxEdge = 0;
            for ( int k = 0; k < 3; ++k )
            {
                maxDist = std::max( maxDist, std::abs( s.dist( d.pts[d.tris[g][k]] ) ) );
                maxEdge = std::max( maxEdge, ( d.pts[d.tris[g][k]] - d.pts[d.tris[g][( k + 1 ) % 3]] ).length() );
            }
            if ( maxDist > 2.5 * d.tol )
                continue;
            const bool sliver = !( maxEdge > 0 ) || 2 * d.areas[g] / maxEdge < 0.5 * d.tol;
            if ( !sliver && std::abs( dot( s.normal( d.centers[g] ), d.normals[g] ) ) < cCos1 )
                continue;
            d.sub[g] = r;
            queue.push_back( g );
        }
    }
}

/// one surface passing through all vertices of two regions, if any
std::optional<Surf> Cleaner::unionFit_( const std::vector<int>& fa, const std::vector<int>& fb, const Region& ra, const Region& rb ) const
{
    const auto& d = d_;
    auto fs = fa;
    fs.insert( fs.end(), fb.begin(), fb.end() );
    const auto pts = d.pointsOf( fs );
    const auto sub = d.pointsOf( fs, 800 );
    std::vector<V3> ns, cs;
    double area = 0;
    for ( int f : fs )
    {
        ns.push_back( d.normals[f] );
        cs.push_back( d.centers[f] );
        area += d.areas[f];
    }
    const double own = ( d.centersDeviation( ra.surf, fa ) * ra.area + d.centersDeviation( rb.surf, fb ) * rb.area ) / area;
    auto accept = [&]( const Surf& q )
    {
        if ( !q.finite() || q.maxDist( pts ) > d.tol )
            return false;
        for ( int f : fs )
            if ( !d.normalsAgree( q, f, true ) )
                return false;
        return d.centersDeviation( q, fs ) <= 2 * own + 1e-4 * d.diag;
    };
    auto tryRefined = [&]( std::optional<Surf> q ) -> std::optional<Surf>
    {
        if ( !q )
            return {};
        q = refine( *q, sub );
        return accept( *q ) ? q : std::nullopt;
    };
    if ( auto q = tryRefined( fitCyl( pts, ns ) ) )
        return q;
    if ( auto q = tryRefined( fitConeByNormals( pts, cs, ns ) ) )
        return q;
    if ( auto q = tryRefined( fitRevolution( pts, cs, ns, SurfType::Cone ) ) )
        return q;
    if ( auto q = tryRefined( fitSphere( pts ) ) )
        return q;
    for ( const auto& ax : axes_ )
        if ( auto q = axisTorus( pts, ax.dir, ax.pos ); q && accept( *q ) )
            return q;
    return tryRefined( fitRevolution( pts, cs, ns, SurfType::Torus ) );
}

/// two smoothly adjacent curved regions that one surface explains are one region,
/// e.g. sectors of a torus that were accepted as short cylinders
void Cleaner::uniteUnderOneSurface_()
{
    auto& d = d_;
    std::set<std::array<int, 4>> tested;
    for ( ;; )
    {
        d.updateStats();
        const auto borders = borders_();
        const auto faces = d.facesOfRegions();
        std::vector<Pair> pairs;
        for ( const auto& [p, b] : borders )
        {
            const auto& ra = d.regions[p.first];
            const auto& rb = d.regions[p.second];
            if ( b.angle() > 15 || ra.surf.type == SurfType::Plane || rb.surf.type == SurfType::Plane || ra.numFaces < 6 || rb.numFaces < 6 )
                continue;
            if ( tested.count( { p.first, p.second, ra.numFaces, rb.numFaces } ) )
                continue;
            pairs.push_back( p );
        }
        std::stable_sort( pairs.begin(), pairs.end(), [&]( const Pair& x, const Pair& y )
        {
            return d.regions[x.first].area + d.regions[x.second].area > d.regions[y.first].area + d.regions[y.second].area;
        } );
        bool done = false;
        for ( const auto& p : pairs )
        {
            auto& ra = d.regions[p.first];
            const auto& rb = d.regions[p.second];
            tested.insert( { p.first, p.second, ra.numFaces, rb.numFaces } );
            if ( auto q = unionFit_( faces[p.first], faces[p.second], ra, rb ) )
            {
                ra.surf = *q;
                moveFaces_( p.second, p.first );
                done = true;
                break;
            }
        }
        if ( !done )
            break;
    }
}

/// a region lying entirely on the surface of a larger adjacent region belongs to it
void Cleaner::absorbLyingOnNeighbour_()
{
    auto& d = d_;
    for ( ;; )
    {
        d.updateStats();
        const auto faces = d.facesOfRegions();
        std::vector<Pair> pairs;
        for ( const auto& [p, b] : borders_() )
        {
            pairs.push_back( p );
            pairs.push_back( { p.second, p.first } );
        }
        std::stable_sort( pairs.begin(), pairs.end(), [&]( const Pair& x, const Pair& y ) { return d.regions[x.first].area > d.regions[y.first].area; } );
        bool done = false;
        for ( const auto& [a, b] : pairs )
        {
            const auto& ra = d.regions[a];
            if ( ra.area <= d.regions[b].area || !strong_( ra ) )
                continue;
            bool ok = true;
            for ( int f : faces[b] )
            {
                for ( int v : d.tris[f] )
                    ok = ok && std::abs( ra.surf.dist( d.pts[v] ) ) <= d.tol;
                ok = ok && std::abs( dot( ra.surf.normal( d.centers[f] ), d.normals[f] ) ) >= cCos40;
                if ( !ok )
                    break;
            }
            if ( !ok )
                continue;
            moveFaces_( b, a );
            done = true;
            break;
        }
        if ( !done )
            break;
    }
}

/// adjacent regions on the same surface are one region
void Cleaner::mergeSameSurface_()
{
    auto& d = d_;
    d.updateStats();
    std::vector<int> parent( d.regions.size() );
    std::iota( parent.begin(), parent.end(), 0 );
    auto find = [&]( int x )
    {
        while ( parent[x] != x )
            x = parent[x] = parent[parent[x]];
        return x;
    };
    std::vector<double> area;
    for ( const auto& r : d.regions )
        area.push_back( r.area );
    for ( const auto& [p, b] : borders_() )
    {
        int ra = find( p.first ), rb = find( p.second );
        if ( ra == rb || !sameSurface( d.regions[p.first].surf, d.regions[p.second].surf, 0.5 * d.tol ) )
            continue;
        if ( area[ra] < area[rb] )
            std::swap( ra, rb );
        parent[rb] = ra;
        area[ra] += area[rb];
    }
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) )
            d.sub[f] = find( d.sub[f] );
}

/// groups of smoothly adjacent weak regions are approximated by B-spline surfaces
void Cleaner::fitFreeform_()
{
    MR_TIMER;
    auto& d = d_;
    d.updateStats();
    const int nr = (int)d.regions.size();
    std::vector<char> weak( nr );
    for ( int i = 0; i < nr; ++i )
        weak[i] = d.regions[i].numFaces > 0 && !strong_( d.regions[i] ) && d.regions[i].surf.type != SurfType::Plane; // one or two triangles can be a true plane
    const auto borders = borders_();
    std::vector<std::vector<int>> smoothNbr( nr );
    for ( const auto& [p, b] : borders )
        if ( b.angle() < 30 )
        {
            smoothNbr[p.first].push_back( p.second );
            smoothNbr[p.second].push_back( p.first );
        }
    std::vector<int> cluster;
    int numClusters = 0;
    auto findClusters = [&]
    {
        cluster.assign( nr, -1 );
        numClusters = 0;
        std::vector<int> stack;
        for ( int r0 = 0; r0 < nr; ++r0 )
        {
            if ( !weak[r0] || cluster[r0] >= 0 )
                continue;
            cluster[r0] = numClusters;
            stack.push_back( r0 );
            while ( !stack.empty() )
            {
                const int x = stack.back();
                stack.pop_back();
                for ( int y : smoothNbr[x] )
                    if ( weak[y] && cluster[y] < 0 )
                    {
                        cluster[y] = numClusters;
                        stack.push_back( y );
                    }
            }
            ++numClusters;
        }
    };
    findClusters();
    std::vector<double> clusterArea( numClusters, 0.0 ), lim( nr, 0.0 ), border( nr, 0.0 );
    for ( int r = 0; r < nr; ++r )
        if ( weak[r] )
            clusterArea[cluster[r]] += d.regions[r].area;
    for ( int r = 0; r < nr; ++r )
        if ( weak[r] )
            lim[r] = 0.03 * clusterArea[cluster[r]];
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) )
            for ( int i = 0; i < 3; ++i )
                if ( const int g = d.nbr[f][i]; g < 0 || d.sub[g] != d.sub[f] )
                    border[d.sub[f]] += ( d.pts[d.tris[f][i]] - d.pts[d.tris[f][( i + 1 ) % 3]] ).length();

    // accidental "strong" regions lying smoothly inside a cluster join it
    for ( ;; )
    {
        std::vector<std::pair<int, double>> add;
        for ( int y = 0; y < nr; ++y )
        {
            const auto& ry = d.regions[y];
            if ( weak[y] || ry.numFaces == 0 || ry.numFaces >= 120 || ry.surf.type == SurfType::Sphere || ry.surf.type == SurfType::BSpline )
                continue;
            double l = 0, shared = 0;
            bool any = false;
            for ( int r : smoothNbr[y] )
                if ( weak[r] )
                {
                    any = true;
                    l = std::max( l, lim[r] );
                    shared += borders.at( std::minmax( y, r ) ).len;
                }
            if ( any && ( ry.area < l || ( ry.area < 15 * l && ( shared > 0.4 * border[y] || !strong_( ry ) ) ) ) )
                add.push_back( { y, l } );
        }
        if ( add.empty() )
            break;
        for ( const auto& [y, l] : add )
        {
            weak[y] = 1;
            lim[y] = l;
        }
    }
    findClusters();

    std::vector<std::vector<int>> clusterFaces( numClusters );
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) && cluster[d.sub[f]] >= 0 )
            clusterFaces[cluster[d.sub[f]]].push_back( f );
    for ( const auto& fs : clusterFaces )
    {
        if ( fs.size() < 30 )
            continue;
        const auto verts = d.vertsOf( fs );
        std::vector<V3> pts;
        pts.reserve( verts.size() );
        for ( int v : verts )
            pts.push_back( d.pts[v] );
        std::vector<Tri> tris;
        tris.reserve( fs.size() );
        for ( int f : fs )
        {
            Tri t;
            for ( int i = 0; i < 3; ++i )
                t[i] = int( std::lower_bound( verts.begin(), verts.end(), d.tris[f][i] ) - verts.begin() );
            tris.push_back( t );
        }
        auto bs = fitBSpline( pts, tris, 0.5 * d.tol );
        if ( !bs )
            continue;
        Region r;
        r.surf.type = SurfType::BSpline;
        r.surf.bs = bs;
        r.seg = d.seg[fs[0]];
        for ( int f : fs )
            d.sub[f] = (int)d.regions.size();
        d.regions.push_back( std::move( r ) );
    }
}

/// regions of (nearly) zero area carry no geometry: give them to the neighbour with the longest common border
void Cleaner::mergeZeroArea_()
{
    auto& d = d_;
    for ( ;; )
    {
        d.updateStats();
        const auto borders = borders_();
        bool done = false;
        for ( int r = 0; r < (int)d.regions.size(); ++r )
        {
            if ( d.regions[r].numFaces == 0 || d.regions[r].area >= 1e-8 * d.total )
                continue;
            int best = -1;
            double bestLen = -1;
            for ( const auto& [p, b] : borders )
                if ( ( p.first == r || p.second == r ) && b.len > bestLen )
                {
                    bestLen = b.len;
                    best = p.first == r ? p.second : p.first;
                }
            if ( best >= 0 )
            {
                moveFaces_( r, best );
                done = true;
                break;
            }
        }
        if ( !done )
            break;
    }
}

/// every region becomes one connected piece, and the regions are renumbered without gaps
void Cleaner::splitOnComponents_()
{
    auto& d = d_;
    std::vector<int> comp( d.nf, -1 );
    std::vector<Region> regions;
    std::vector<int> stack;
    for ( int f0 = 0; f0 < d.nf; ++f0 )
    {
        if ( !d.valid( f0 ) || comp[f0] >= 0 )
            continue;
        const int id = (int)regions.size();
        regions.push_back( d.regions[d.sub[f0]] );
        comp[f0] = id;
        stack.push_back( f0 );
        while ( !stack.empty() )
        {
            const int f = stack.back();
            stack.pop_back();
            for ( int g : d.nbr[f] )
                if ( g >= 0 && comp[g] < 0 && d.sub[g] == d.sub[f] )
                {
                    comp[g] = id;
                    stack.push_back( g );
                }
        }
    }
    d.sub = std::move( comp );
    d.regions = std::move( regions );
    d.updateStats();
}

void Cleaner::run()
{
    MR_TIMER;
    auto& d = d_;
    d.updateStats();
    for ( const auto& r : d.regions )
        if ( ( r.surf.type == SurfType::Cyl || r.surf.type == SurfType::Cone ) && r.numFaces >= 20 && r.area >= 2e-4 * d.total )
            addAxis( axes_, r.surf, d.tol );
    extendStrong_();
    uniteUnderOneSurface_();
    absorbLyingOnNeighbour_();
    mergeSameSurface_();
    fitFreeform_();
    mergeZeroArea_();
    splitOnComponents_();
}

/// a curve of the boundary between two regions
struct Curve
{
    enum class Kind { Line, Circle, BSpline } kind = Kind::Line;
    V3 c, n, x;   ///< circle: center, normal and the direction to the start point
    double r = 0; ///< circle radius
    int deg = 0;  ///< B-spline degree
    std::vector<double> knots;
    std::vector<V3> cp;
    double dev = 0; ///< maximum distance from B-spline to the surfaces
};

/// a part of the boundary between two regions: one curve between two vertices
struct BRepEdge
{
    std::vector<int> verts; ///< the chain of mesh vertices
    Curve curve;
};

/// creates boundary representation of the regions, and writes it in STEP format
class BRepWriter
{
public:
    BRepWriter( const Data& d, const std::string& name ) : d_( d ), name_( name ) {}
    MeshToStepStats write( std::ostream& out );

private:
    using Surfs = std::vector<const Surf*>;
    using DirEdge = std::pair<int, int>;
    struct EdgeUse
    {
        int edge = -1;
        bool forward = true;
        bool first = false; ///< it is the first mesh edge of BRepEdge passed in this direction
    };

    double curveTol_( const Surfs& ss ) const;
    double onSurfs_( const std::vector<V3>& pts, const Surfs& ss ) const;
    std::vector<V3> pointsOf_( const std::vector<int>& ids ) const;
    std::optional<Curve> tryLine_( const std::vector<V3>& p, const Surfs& ss ) const;
    std::optional<Curve> circleFrom_( const std::vector<V3>& p, const V3& ctr, V3 n, double r, const Surfs& ss, bool closed ) const;
    std::optional<Curve> axisCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const;
    std::optional<Curve> tryCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const;
    Curve spline_( const std::vector<V3>& p, const Surfs& ss ) const;
    void splitFit_( const std::vector<int>& ids, const Surfs& ss );
    void findEdges_();
    void findLoops_();

    std::string add_( std::string s );
    std::string point_( const V3& p );
    std::string dir_( const V3& d );
    std::string axis2_( const V3& p, const V3& z, const V3* x = nullptr );
    std::string vertex_( int v );

    const Data& d_;
    std::string name_;
    std::map<DirEdge, std::pair<int, int>> bnd_; ///< boundary mesh edge (u,v) -> its left face and the index of the edge in it
    std::vector<BRepEdge> edges_;
    std::map<DirEdge, EdgeUse> edgeOf_;
    std::vector<std::vector<std::vector<std::pair<int, bool>>>> loops_; ///< region -> loops -> (edge, forward)
    std::vector<std::string> lines_;
    std::map<int, std::string> vertexIds_;
};

double BRepWriter::curveTol_( const Surfs& ss ) const
{
    double res = d_.tol;
    for ( auto s : ss )
        if ( s->type == SurfType::BSpline )
            res = std::max( res, 3 * s->bs->err );
    return res;
}

double BRepWriter::onSurfs_( const std::vector<V3>& pts, const Surfs& ss ) const
{
    double res = 0;
    for ( auto s : ss )
        res = std::max( res, s->maxDist( pts ) );
    return res;
}

std::vector<V3> BRepWriter::pointsOf_( const std::vector<int>& ids ) const
{
    std::vector<V3> res;
    res.reserve( ids.size() );
    for ( int v : ids )
        res.push_back( d_.pts[v] );
    return res;
}

std::optional<Curve> BRepWriter::tryLine_( const std::vector<V3>& p, const Surfs& ss ) const
{
    V3 dir = p.back() - p.front();
    const double len = dir.length();
    if ( len < 1e-9 * d_.diag )
        return {};
    dir /= len;
    for ( const auto& q : p )
    {
        const V3 r = q - p.front();
        if ( ( r - dir * dot( r, dir ) ).length() > 0.5 * d_.tol )
            return {};
    }
    std::vector<V3> samples;
    for ( int i = 0; i <= 8; ++i )
        samples.push_back( p.front() + ( p.back() - p.front() ) * ( i / 8.0 ) );
    if ( onSurfs_( samples, ss ) > curveTol_( ss ) )
        return {};
    return Curve{};
}

std::optional<Curve> BRepWriter::circleFrom_( const std::vector<V3>& p, const V3& ctr, V3 n, double r, const Surfs& ss, bool closed ) const
{
    // orient the axis so that the chain runs counter-clockwise
    V3 tot;
    for ( size_t i = 0; i + 1 < p.size(); ++i )
        tot += cross( p[i] - ctr, p[i + 1] - ctr );
    if ( dot( tot, n ) < 0 )
        n = -n;
    V3 xd = p[0] - ctr;
    xd -= n * dot( xd, n );
    if ( !( xd.length() > 0 ) )
        return {};
    xd = xd.normalized();
    const V3 yd = cross( n, xd );
    double prev = 0;
    for ( size_t i = 1; i < p.size(); ++i )
    {
        const V3 d = p[i] - ctr;
        double ang = std::atan2( dot( d, yd ), dot( d, xd ) );
        while ( ang < prev - PI )
            ang += 2 * PI;
        while ( ang > prev + PI )
            ang -= 2 * PI;
        if ( closed && i + 1 == p.size() )
            ang = 2 * PI;
        if ( ang <= prev )
            return {};
        prev = ang;
    }
    if ( prev > 2 * PI + 1e-9 )
        return {};
    std::vector<V3> samples;
    const int num = 4 * (int)p.size();
    for ( int i = 0; i <= num; ++i )
    {
        const double t = prev * i / num;
        samples.push_back( ctr + r * ( std::cos( t ) * xd + std::sin( t ) * yd ) );
    }
    if ( onSurfs_( samples, ss ) > curveTol_( ss ) )
        return {};
    Curve res;
    res.kind = Curve::Kind::Circle;
    res.c = ctr;
    res.n = n;
    res.x = xd;
    res.r = r;
    return res;
}

/// circle around the axis of an adjacent surface of revolution: steadier than a free fit to few points
std::optional<Curve> BRepWriter::axisCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const
{
    for ( auto s : ss )
    {
        if ( s->type != SurfType::Cyl && s->type != SurfType::Cone && s->type != SurfType::Torus )
            continue;
        double minH = DBL_MAX, maxH = -DBL_MAX, minR = DBL_MAX, maxR = -DBL_MAX, sumH = 0, sumR = 0;
        for ( const auto& q : p )
        {
            const V3 d = q - s->c;
            const double h = dot( d, s->a );
            const double rho = ( d - h * s->a ).length();
            minH = std::min( minH, h );
            maxH = std::max( maxH, h );
            minR = std::min( minR, rho );
            maxR = std::max( maxR, rho );
            sumH += h;
            sumR += rho;
        }
        const double rad = sumR / double( p.size() );
        if ( maxH - minH > d_.tol || maxR - minR > d_.tol || rad < 1e-9 * d_.diag )
            continue;
        if ( auto res = circleFrom_( p, s->c + s->a * ( sumH / double( p.size() ) ), s->a, rad, ss, closed ) )
            return res;
    }
    return {};
}

std::optional<Curve> BRepWriter::tryCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const
{
    std::vector<V3> q( p.begin(), closed ? p.end() - 1 : p.end() );
    if ( q.size() < 3 )
        return {};
    const V3 c0 = centroid( q );
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for ( const auto& pt : q )
    {
        const V3 d = pt - c0;
        const Eigen::Vector3d e( d.x, d.y, d.z );
        cov += e * e.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es( cov );
    if ( std::sqrt( std::max( es.eigenvalues()[1], 0.0 ) ) < 1e-6 * std::sqrt( std::max( es.eigenvalues()[2], 1e-24 ) ) )
        return {};
    auto col = [&]( int i ) { return V3( es.eigenvectors()( 0, i ), es.eigenvectors()( 1, i ), es.eigenvectors()( 2, i ) ); };
    const V3 n = col( 0 ), u = col( 2 ), v = col( 1 );
    std::vector<double> xs, ys;
    for ( const auto& pt : q )
    {
        if ( std::abs( dot( pt - c0, n ) ) > 0.5 * d_.tol )
            return {};
        xs.push_back( dot( pt - c0, u ) );
        ys.push_back( dot( pt - c0, v ) );
    }
    double cx, cy, r;
    if ( !fitCircle2( xs, ys, cx, cy, r ) || r < 1e-9 * d_.diag )
        return {};
    const V3 ctr = c0 + cx * u + cy * v;
    for ( const auto& pt : q )
        if ( std::abs( ( pt - ctr ).length() - r ) > 0.5 * d_.tol )
            return {};
    return circleFrom_( p, ctr, n, r, ss, closed );
}

/// B-spline curve interpolating given points and additional points between them projected on the surfaces
Curve BRepWriter::spline_( const std::vector<V3>& p, const Surfs& ss ) const
{
    auto q = p;
    for ( int iter = 0; iter < 2; ++iter )
    {
        std::vector<V3> z;
        z.reserve( 2 * q.size() );
        for ( size_t i = 0; i + 1 < q.size(); ++i )
        {
            z.push_back( q[i] );
            V3 m = ( q[i] + q[i + 1] ) / 2.0;
            for ( int it = 0; it < 30; ++it )
                for ( auto s : ss )
                    m -= s->dist( m ) * s->normal( m );
            z.push_back( m );
        }
        z.push_back( q.back() );
        q = std::move( z );
    }
    const int n = (int)q.size();
    auto project = [&]( V3 m )
    {
        for ( int it = 0; it < 30; ++it )
            for ( auto s : ss )
                m -= s->dist( m ) * s->normal( m );
        return m;
    };
    {
        // evenly spaced points make the interpolation stable
        std::vector<double> len( n, 0.0 );
        for ( int i = 1; i < n; ++i )
            len[i] = len[i - 1] + ( q[i] - q[i - 1] ).length();
        std::vector<V3> even( n );
        even.front() = q.front();
        even.back() = q.back();
        for ( int i = 1, j = 1; i + 1 < n; ++i )
        {
            const double l = len.back() * i / ( n - 1 );
            while ( j + 1 < n && len[j] < l )
                ++j;
            const double seg = len[j] - len[j - 1];
            const double a = seg > 0 ? ( l - len[j - 1] ) / seg : 0;
            even[i] = project( q[j - 1] * ( 1 - a ) + q[j] * a );
        }
        q = std::move( even );
    }
    std::vector<double> t( n, 0.0 );
    for ( int i = 1; i < n; ++i )
        t[i] = t[i - 1] + ( q[i] - q[i - 1] ).length();
    for ( auto& x : t )
        x /= std::max( t.back(), 1e-300 );
    t.back() = 1;

    Curve res;
    res.kind = Curve::Kind::BSpline;
    res.deg = std::min( 3, n - 1 );
    const int k = res.deg;
    res.knots.assign( n + k + 1, 1.0 );
    for ( int i = 0; i <= k; ++i )
        res.knots[i] = 0;
    for ( int j = 1; j < n - k; ++j )
        res.knots[j + k] = std::accumulate( t.begin() + j, t.begin() + j + k, 0.0 ) / k;
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero( n, n );
    Eigen::MatrixXd rhs( n, 3 );
    for ( int i = 0; i < n; ++i )
    {
        double b[4];
        const int i0 = bsplineBasis( res.knots, k, n, t[i], b, nullptr );
        for ( int j = 0; j <= k; ++j )
            m( i, i0 + j ) = b[j];
        for ( int c = 0; c < 3; ++c )
            rhs( i, c ) = q[i][c];
    }
    const Eigen::MatrixXd sol = m.partialPivLu().solve( rhs );
    res.cp.resize( n );
    for ( int i = 0; i < n; ++i )
        res.cp[i] = V3( sol( i, 0 ), sol( i, 1 ), sol( i, 2 ) );

    std::vector<V3> samples;
    const int num = 8 * n;
    for ( int i = 0; i < num; ++i )
    {
        double b[4];
        const int i0 = bsplineBasis( res.knots, k, n, double( i ) / ( num - 1 ), b, nullptr );
        V3 s;
        for ( int j = 0; j <= k; ++j )
            s += b[j] * res.cp[i0 + j];
        samples.push_back( s );
    }
    res.dev = onSurfs_( samples, ss );
    if ( !std::isfinite( res.dev ) )
        res.dev = DBL_MAX;
    return res;
}

/// approximates the chain of mesh vertices by a line or a circle if possible, otherwise splits it in the corners and approximates by splines
void BRepWriter::splitFit_( const std::vector<int>& ids, const Surfs& ss )
{
    const auto p = pointsOf_( ids );
    const bool closed = ids.front() == ids.back();
    std::optional<Curve> c;
    if ( !closed )
        c = tryLine_( p, ss );
    if ( !c )
        c = axisCircle_( p, ss, closed );
    if ( !c )
        c = tryCircle_( p, ss, closed );
    if ( c )
    {
        edges_.push_back( { ids, *c } );
        return;
    }
    static const double cCorner = 22 * PI / 180;
    const int n = (int)ids.size();
    auto sub = [&]( int from, int to ) { return std::vector<int>( ids.begin() + from, ids.begin() + to + 1 ); };
    if ( n > 2 )
    {
        std::vector<V3> dirs;
        for ( int i = 0; i + 1 < n; ++i )
            dirs.push_back( ( p[i + 1] - p[i] ).normalized() );
        auto angle = [&]( int i, int j ) { return std::acos( std::clamp( dot( dirs[i], dirs[j] ), -1.0, 1.0 ) ); };
        if ( closed )
        {
            // angle[0] is in the start vertex, angle[i] is in vertex i
            int best = 0;
            double bestAng = angle( n - 2, 0 );
            for ( int i = 1; i + 1 < n; ++i )
                if ( double a = angle( i - 1, i ); a > bestAng )
                {
                    bestAng = a;
                    best = i;
                }
            if ( bestAng > cCorner && best != 0 )
            {
                // restart the closed chain in its sharpest corner
                std::vector<int> rotated( ids.begin() + best, ids.end() - 1 );
                rotated.insert( rotated.end(), ids.begin(), ids.begin() + best + 1 );
                return splitFit_( rotated, ss );
            }
            if ( bestAng <= cCorner )
            {
                // smooth closed non-circular curve: cut it in two halves
                const int h = n / 2;
                splitFit_( sub( 0, h ), ss );
                splitFit_( sub( h, n - 1 ), ss );
                return;
            }
        }
        std::vector<int> corners;
        for ( int i = 1; i + 1 < n; ++i )
            if ( angle( i - 1, i ) > cCorner )
                corners.push_back( i );
        if ( !corners.empty() )
        {
            corners.push_back( n - 1 );
            int prev = 0;
            for ( int i : corners )
            {
                splitFit_( sub( prev, i ), ss );
                prev = i;
            }
            return;
        }
    }
    Curve sp = spline_( p, ss );
    if ( n == 2 && sp.dev >= curveTol_( ss ) )
        sp = Curve{}; // no better curve than a line segment
    edges_.push_back( { ids, std::move( sp ) } );
}

void BRepWriter::findEdges_()
{
    MR_TIMER;
    const auto& d = d_;
    std::map<int, std::set<int>> adj;
    for ( int f = 0; f < d.nf; ++f )
    {
        if ( !d.valid( f ) )
            continue;
        for ( int i = 0; i < 3; ++i )
        {
            const int g = d.nbr[f][i];
            if ( g >= 0 && d.sub[g] == d.sub[f] )
                continue;
            const int u = d.tris[f][i], v = d.tris[f][( i + 1 ) % 3];
            bnd_[{ u, v }] = { f, i };
            adj[u].insert( v );
            adj[v].insert( u );
        }
    }
    auto isNode = [&]( int v ) { return adj[v].size() != 2; };
    std::set<DirEdge> visited;
    auto walk = [&]( int u, int v )
    {
        std::vector<int> vs{ u, v };
        while ( !isNode( vs.back() ) && vs.back() != vs.front() )
        {
            const auto& s = adj[vs.back()];
            const int prev = vs[vs.size() - 2];
            vs.push_back( *s.begin() == prev ? *s.rbegin() : *s.begin() );
        }
        for ( size_t i = 0; i + 1 < vs.size(); ++i )
        {
            visited.insert( { vs[i], vs[i + 1] } );
            visited.insert( { vs[i + 1], vs[i] } );
        }
        // the surfaces on both sides of the chain
        Surfs ss;
        for ( const DirEdge& de : { DirEdge{ u, v }, DirEdge{ v, u } } )
            if ( auto it = bnd_.find( de ); it != bnd_.end() )
                ss.push_back( &d.regions[d.sub[it->second.first]].surf );
        const size_t first = edges_.size();
        splitFit_( vs, ss );
        for ( size_t e = first; e < edges_.size(); ++e )
        {
            const auto& ids = edges_[e].verts;
            for ( size_t i = 0; i + 1 < ids.size(); ++i )
            {
                edgeOf_[{ ids[i], ids[i + 1] }] = { (int)e, true, i == 0 };
                edgeOf_[{ ids[i + 1], ids[i] }] = { (int)e, false, i + 2 == ids.size() };
            }
        }
    };
    for ( const auto& [u, s] : adj )
        if ( s.size() != 2 )
            for ( int v : std::vector<int>( s.begin(), s.end() ) )
                if ( !visited.count( { u, v } ) )
                    walk( u, v );
    // closed chains without nodes
    for ( const auto& [de, fi] : std::map<DirEdge, std::pair<int, int>>( bnd_ ) )
        if ( !visited.count( de ) )
            walk( de.first, de.second );
}

void BRepWriter::findLoops_()
{
    const auto& d = d_;
    loops_.resize( d.regions.size() );
    std::set<DirEdge> done;
    for ( const auto& [start, fi] : bnd_ )
    {
        if ( done.count( start ) )
            continue;
        const int r = d.sub[fi.first];
        std::vector<std::pair<int, bool>> loop;
        for ( DirEdge cur = start; done.insert( cur ).second; )
        {
            const auto& use = edgeOf_.at( cur );
            if ( use.first )
                loop.push_back( { use.edge, use.forward } );
            // the next boundary edge of the same region: rotate around the destination vertex
            int f = bnd_.at( cur ).first;
            for ( ;; )
            {
                const int i = int( std::find( d.tris[f].begin(), d.tris[f].end(), cur.second ) - d.tris[f].begin() );
                const int g = d.nbr[f][i];
                if ( g < 0 || d.sub[g] != r )
                {
                    cur = { cur.second, d.tris[f][( i + 1 ) % 3] };
                    break;
                }
                f = g;
            }
        }
        loops_[r].push_back( std::move( loop ) );
    }
}

std::string BRepWriter::add_( std::string s )
{
    lines_.push_back( std::move( s ) );
    return fmt::format( "#{}", lines_.size() );
}

/// real number in the form required by STEP: with a decimal point
std::string stepReal( double x )
{
    if ( !std::isfinite( x ) )
        x = 0;
    std::string s = fmt::format( "{}", x );
    const auto e = s.find( 'e' );
    if ( e == std::string::npos )
    {
        if ( s.find( '.' ) == std::string::npos )
            s += '.';
        return s;
    }
    std::string mant = s.substr( 0, e );
    if ( mant.find( '.' ) == std::string::npos )
        mant += '.';
    return mant + "E" + std::to_string( std::stoi( s.substr( e + 1 ) ) );
}

double rounded( double x, double mult )
{
    return std::round( x * mult ) / mult + 0.0;
}

std::string BRepWriter::point_( const V3& p )
{
    return add_( fmt::format( "CARTESIAN_POINT('',({},{},{}))", stepReal( rounded( p.x, 1e9 ) ), stepReal( rounded( p.y, 1e9 ) ), stepReal( rounded( p.z, 1e9 ) ) ) );
}

std::string BRepWriter::dir_( const V3& d0 )
{
    const V3 d = d0.normalized();
    return add_( fmt::format( "DIRECTION('',({},{},{}))", stepReal( rounded( d.x, 1e12 ) ), stepReal( rounded( d.y, 1e12 ) ), stepReal( rounded( d.z, 1e12 ) ) ) );
}

std::string BRepWriter::axis2_( const V3& p, const V3& z0, const V3* x0 )
{
    const V3 z = z0.normalized();
    V3 x, y;
    if ( x0 )
        x = *x0;
    else
        frame( z, x, y );
    x -= z * dot( x, z );
    const auto ps = point_( p );
    const auto zs = dir_( z );
    return add_( fmt::format( "AXIS2_PLACEMENT_3D('',{},{},{})", ps, zs, dir_( x ) ) );
}

std::string BRepWriter::vertex_( int v )
{
    auto [it, inserted] = vertexIds_.insert( { v, {} } );
    if ( inserted )
        it->second = add_( fmt::format( "VERTEX_POINT('',{})", point_( d_.pts[v] ) ) );
    return it->second;
}

/// makes the direction numerically parallel to a coordinate axis exactly parallel
V3 snapped( const V3& a )
{
    for ( int i = 0; i < 3; ++i )
        if ( std::abs( std::abs( a[i] ) - 1 ) < 1e-9 )
        {
            V3 res;
            res[i] = a[i] > 0 ? 1 : -1;
            return res;
        }
    return a;
}

std::string joined( const std::vector<std::string>& v )
{
    std::string res;
    for ( const auto& s : v )
    {
        if ( !res.empty() )
            res += ',';
        res += s;
    }
    return res;
}

/// distinct knots and their multiplicities as two STEP lists
std::pair<std::string, std::string> knotLists( const std::vector<double>& knots )
{
    std::vector<std::string> ks, ms;
    for ( size_t i = 0; i < knots.size(); )
    {
        size_t j = i;
        while ( j < knots.size() && knots[j] == knots[i] )
            ++j;
        ks.push_back( stepReal( knots[i] ) );
        ms.push_back( std::to_string( j - i ) );
        i = j;
    }
    return { joined( ms ), joined( ks ) };
}

MeshToStepStats BRepWriter::write( std::ostream& out )
{
    MR_TIMER;
    const auto& d = d_;
    findEdges_();
    findLoops_();

    const auto app = add_( "APPLICATION_CONTEXT('automotive design')" );
    add_( fmt::format( "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2010,{})", app ) );
    const auto pc = add_( fmt::format( "PRODUCT_CONTEXT('',{},'mechanical')", app ) );
    const auto prod = add_( fmt::format( "PRODUCT('{0}','{0}','',({1}))", name_, pc ) );
    const auto pdf = add_( fmt::format( "PRODUCT_DEFINITION_FORMATION('','',{})", prod ) );
    const auto pdc = add_( fmt::format( "PRODUCT_DEFINITION_CONTEXT('part definition',{},'design')", app ) );
    const auto pd = add_( fmt::format( "PRODUCT_DEFINITION('design','',{},{})", pdf, pdc ) );
    const auto pds = add_( fmt::format( "PRODUCT_DEFINITION_SHAPE('','',{})", pd ) );
    add_( fmt::format( "PRODUCT_RELATED_PRODUCT_CATEGORY('part','',({}))", prod ) );
    const auto lu = add_( "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))" );
    const auto au = add_( "(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))" );
    const auto su = add_( "(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())" );
    const auto unc = add_( fmt::format( "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({}),{},'distance_accuracy_value','')", stepReal( d.tol ), lu ) );
    const auto ctx = add_( fmt::format( "(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({}))GLOBAL_UNIT_ASSIGNED_CONTEXT(({},{},{}))REPRESENTATION_CONTEXT('',''))", unc, lu, au, su ) );

    std::vector<std::string> edgeIds;
    for ( const auto& e : edges_ )
    {
        const auto& cv = e.curve;
        const V3& p0 = d.pts[e.verts.front()];
        const V3& p1 = d.pts[e.verts.back()];
        std::string geom;
        if ( cv.kind == Curve::Kind::Line )
        {
            const auto ps = point_( p0 );
            const auto vec = add_( fmt::format( "VECTOR('',{},{})", dir_( p1 - p0 ), stepReal( ( p1 - p0 ).length() ) ) );
            geom = add_( fmt::format( "LINE('',{},{})", ps, vec ) );
        }
        else if ( cv.kind == Curve::Kind::Circle )
            geom = add_( fmt::format( "CIRCLE('',{},{})", axis2_( cv.c, snapped( cv.n ), &cv.x ), stepReal( rounded( cv.r, 1e9 ) ) ) );
        else
        {
            std::vector<std::string> cps;
            for ( const auto& p : cv.cp )
                cps.push_back( point_( p ) );
            const auto [mults, knots] = knotLists( cv.knots );
            geom = add_( fmt::format( "B_SPLINE_CURVE_WITH_KNOTS('',{},({}),.UNSPECIFIED.,.F.,.F.,({}),({}),.UNSPECIFIED.)", cv.deg, joined( cps ), mults, knots ) );
        }
        const auto v0 = vertex_( e.verts.front() );
        const auto v1 = vertex_( e.verts.back() );
        edgeIds.push_back( add_( fmt::format( "EDGE_CURVE('',{},{},{},.T.)", v0, v1, geom ) ) );
    }

    MeshToStepStats stats;
    const auto regFaces = d.facesOfRegions();
    std::vector<std::string> faceIds( d.regions.size() );
    for ( size_t r = 0; r < d.regions.size(); ++r )
    {
        const Surf& s = d.regions[r].surf;
        const auto& fs = regFaces[r];
        if ( fs.empty() )
            continue;
        const auto verts = d.vertsOf( fs );
        double minH = DBL_MAX, maxH = -DBL_MAX;
        for ( int v : verts )
        {
            const double h = dot( d.pts[v] - s.c, s.a );
            minH = std::min( minH, h );
            maxH = std::max( maxH, h );
        }
        std::string srf;
        switch ( s.type )
        {
        case SurfType::Plane:
            ++stats.planes;
            srf = add_( fmt::format( "PLANE('',{})", axis2_( s.c, snapped( s.a ) ) ) );
            break;
        case SurfType::Cyl:
            ++stats.cylinders;
            srf = add_( fmt::format( "CYLINDRICAL_SURFACE('',{},{})", axis2_( s.c + s.a * minH, snapped( s.a ) ), stepReal( rounded( s.r, 1e9 ) ) ) );
            break;
        case SurfType::Cone:
        {
            ++stats.cones;
            const double h0 = ( minH + maxH ) / 2;
            srf = add_( fmt::format( "CONICAL_SURFACE('',{},{},{})", axis2_( s.c + s.a * h0, snapped( s.a ) ), stepReal( std::abs( h0 ) * std::tan( s.r ) ), stepReal( s.r ) ) );
            break;
        }
        case SurfType::Sphere:
            ++stats.spheres;
            srf = add_( fmt::format( "SPHERICAL_SURFACE('',{},{})", axis2_( s.c, V3( 0, 0, 1 ) ), stepReal( rounded( s.r, 1e9 ) ) ) );
            break;
        case SurfType::Torus:
            ++stats.tori;
            srf = add_( fmt::format( "TOROIDAL_SURFACE('',{},{},{})", axis2_( s.c, snapped( s.a ) ), stepReal( rounded( s.R, 1e9 ) ), stepReal( rounded( s.r, 1e9 ) ) ) );
            break;
        case SurfType::BSpline:
        {
            ++stats.bsplines;
            stats.maxBSplineError = std::max( stats.maxBSplineError, float( s.bs->err ) );
            std::vector<std::string> rows;
            for ( int i = 0; i < s.bs->nu; ++i )
            {
                std::vector<std::string> row;
                for ( int j = 0; j < s.bs->nv; ++j )
                    row.push_back( point_( s.bs->cp[i * s.bs->nv + j] ) );
                rows.push_back( fmt::format( "({})", joined( row ) ) );
            }
            const auto [mu, ku] = knotLists( s.bs->tu );
            const auto [mv, kv] = knotLists( s.bs->tv );
            srf = add_( fmt::format( "B_SPLINE_SURFACE_WITH_KNOTS('',{0},{0},({1}),.UNSPECIFIED.,.F.,.F.,.F.,({2}),({3}),({4}),({5}),.UNSPECIFIED.)",
                BSplineSurf::deg, joined( rows ), mu, mv, ku, kv ) );
            break;
        }
        }
        double sense = 0;
        for ( int f : fs )
            sense += dot( s.normal( d.centers[f] ), d.normals[f] ) * d.areas[f];

        // the loop with the largest bounding box is the outer one
        const auto& loops = loops_[r];
        std::vector<double> extent;
        std::set<int> loopVerts;
        for ( const auto& loop : loops )
        {
            Box3d box;
            for ( const auto& [e, fwd] : loop )
                for ( int v : edges_[e].verts )
                {
                    box.include( d.pts[v] );
                    loopVerts.insert( v );
                }
            extent.push_back( box.valid() ? box.diagonal() : 0 );
        }
        std::vector<int> order( loops.size() );
        std::iota( order.begin(), order.end(), 0 );
        std::stable_sort( order.begin(), order.end(), [&]( int x, int y ) { return extent[x] > extent[y]; } );
        std::vector<std::string> bounds;
        for ( int li : order )
        {
            std::vector<std::string> oes;
            for ( const auto& [e, fwd] : loops[li] )
                oes.push_back( add_( fmt::format( "ORIENTED_EDGE('',*,*,{},{})", edgeIds[e], fwd ? ".T." : ".F." ) ) );
            const auto el = add_( fmt::format( "EDGE_LOOP('',({}))", joined( oes ) ) );
            bounds.push_back( add_( fmt::format( "{}('',{},.T.)", bounds.empty() && s.type == SurfType::Plane ? "FACE_OUTER_BOUND" : "FACE_BOUND", el ) ) );
        }
        auto addVertexLoop = [&]( int v )
        {
            const auto vl = add_( fmt::format( "VERTEX_LOOP('',{})", vertex_( v ) ) );
            bounds.push_back( add_( fmt::format( "FACE_BOUND('',{},.T.)", vl ) ) );
        };
        if ( s.type == SurfType::Cone )
        {
            // the apex inside the face
            int best = verts[0];
            for ( int v : verts )
                if ( ( d.pts[v] - s.c ).lengthSq() < ( d.pts[best] - s.c ).lengthSq() )
                    best = v;
            if ( ( d.pts[best] - s.c ).length() < 5 * d.tol && !loopVerts.count( best ) )
                addVertexLoop( best );
        }
        if ( bounds.empty() )
            addVertexLoop( verts[0] );
        faceIds[r] = add_( fmt::format( "ADVANCED_FACE('',({}),{},{})", joined( bounds ), srf, sense > 0 ? ".T." : ".F." ) );
    }

    // every connected component of the mesh makes its own shell
    std::vector<int> comp( d.nf, -1 );
    std::vector<std::string> shells;
    bool allClosed = true;
    std::vector<int> stack;
    for ( int f0 = 0; f0 < d.nf; ++f0 )
    {
        if ( !d.valid( f0 ) || comp[f0] >= 0 )
            continue;
        const int id = (int)shells.size();
        std::set<int> regs;
        bool closed = true;
        comp[f0] = id;
        stack.push_back( f0 );
        while ( !stack.empty() )
        {
            const int f = stack.back();
            stack.pop_back();
            regs.insert( d.sub[f] );
            for ( int g : d.nbr[f] )
            {
                if ( g < 0 )
                    closed = false;
                else if ( comp[g] < 0 )
                {
                    comp[g] = id;
                    stack.push_back( g );
                }
            }
        }
        std::vector<std::string> fids;
        for ( int r : regs )
            fids.push_back( faceIds[r] );
        shells.push_back( add_( fmt::format( "{}('',({}))", closed ? "CLOSED_SHELL" : "OPEN_SHELL", joined( fids ) ) ) );
        allClosed = allClosed && closed;
    }
    const V3 xAxis( 1, 0, 0 );
    const auto origin = axis2_( V3(), V3( 0, 0, 1 ), &xAxis );
    std::string rep;
    if ( allClosed )
    {
        std::vector<std::string> items{ origin };
        for ( const auto& sh : shells )
            items.push_back( add_( fmt::format( "MANIFOLD_SOLID_BREP('{}',{})", name_, sh ) ) );
        rep = add_( fmt::format( "ADVANCED_BREP_SHAPE_REPRESENTATION('',({}),{})", joined( items ), ctx ) );
    }
    else
    {
        const auto model = add_( fmt::format( "SHELL_BASED_SURFACE_MODEL('{}',({}))", name_, joined( shells ) ) );
        rep = add_( fmt::format( "MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',({},{}),{})", origin, model, ctx ) );
    }
    add_( fmt::format( "SHAPE_DEFINITION_REPRESENTATION({},{})", pds, rep ) );

    out << "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('surfaces fitted to a mesh'),'2;1');\n"
        << "FILE_NAME('" << name_ << "','',(''),(''),'MeshLib','MeshLib','');\n"
        << "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 }'));\nENDSEC;\nDATA;\n";
    for ( size_t i = 0; i < lines_.size(); ++i )
        out << '#' << i + 1 << '=' << lines_[i] << ";\n";
    out << "ENDSEC;\nEND-ISO-10303-21;\n";
    return stats;
}

} //anonymous namespace

Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, std::ostream& out, const MeshToStepSettings& settings )
{
    MR_TIMER;
    if ( mesh.topology.numValidFaces() <= 0 )
        return unexpected( "Mesh is empty" );
    auto d = makeData( mesh, settings );
    if ( !( d.tol > 0 ) )
        return unexpected( "Mesh is degenerate" );
    if ( !Grower( d, subprogress( settings.progress, 0.0f, 0.8f ) ).run() )
        return unexpectedOperationCanceled();
    Cleaner( d ).run();
    if ( !reportProgress( settings.progress, 0.9f ) )
        return unexpectedOperationCanceled();
    std::string name;
    for ( char c : settings.name )
        name += c == '\'' || c == '\\' || (unsigned char)c < 32 ? '_' : c;
    const auto res = BRepWriter( d, name ).write( out );
    if ( !out )
        return unexpected( "Stream write error" );
    reportProgress( settings.progress, 1.0f );
    return res;
}

Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, const std::filesystem::path& file, const MeshToStepSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( "Cannot open file for writing " + utf8string( file ) );
    return saveMeshAsStep( mesh, out, settings );
}

} //namespace MR
