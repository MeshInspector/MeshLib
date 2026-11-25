#include "MRBSpline.h"
#include "MRSymMatrix2.h"
#include "MRPch/MREigenCore.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <algorithm>
#include <cassert>
#include <numeric>

namespace MR
{

namespace
{

constexpr int cMaxDegree = 7;

/// values of (deg+1) non-zero B-spline basis functions of n control points at t, and optionally their derivatives;
/// \return the index of the first of them
int bsplineBasis( const std::vector<double>& knots, int deg, int n, double t, double* vals, double* ders )
{
    assert( deg >= 1 && deg <= cMaxDegree );
    int span = deg;
    while ( span + 1 < n && knots[span + 1] <= t )
        ++span;
    double left[cMaxDegree + 1], right[cMaxDegree + 1], low[cMaxDegree + 1];
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

/// clamped knots with uniform inner spacing on [0, 1] for given number of control points and degree
std::vector<double> makeClampedUniformKnots( int numControlPoints, int degree )
{
    const int n = numControlPoints;
    std::vector<double> res( n + degree + 1, 1.0 );
    for ( int i = 0; i <= degree; ++i )
        res[i] = 0;
    for ( int i = degree + 1; i < n; ++i )
        res[i] = double( i - degree ) / ( n - degree );
    return res;
}

} //anonymous namespace

Vector3d BSplineCurve::eval( double t ) const
{
    double b[cMaxDegree + 1];
    const int i0 = bsplineBasis( knots, degree, int( controlPoints.size() ), t, b, nullptr );
    Vector3d res;
    for ( int j = 0; j <= degree; ++j )
        res += b[j] * controlPoints[i0 + j];
    return res;
}

Vector3d BSplineSurface::eval( const Vector2d& uv, Vector3d* du, Vector3d* dv ) const
{
    double bu[cMaxDegree + 1], bv[cMaxDegree + 1], dbu[cMaxDegree + 1], dbv[cMaxDegree + 1];
    const int nv = numV();
    const int iu = bsplineBasis( knotsU, degreeU, numU(), uv.x, bu, du ? dbu : nullptr );
    const int iv = bsplineBasis( knotsV, degreeV, nv, uv.y, bv, dv ? dbv : nullptr );
    Vector3d res;
    if ( du )
        *du = Vector3d();
    if ( dv )
        *dv = Vector3d();
    for ( int a = 0; a <= degreeU; ++a )
        for ( int b = 0; b <= degreeV; ++b )
        {
            const Vector3d& p = controlPoints[( iu + a ) * nv + iv + b];
            res += bu[a] * bv[b] * p;
            if ( du )
                *du += dbu[a] * bv[b] * p;
            if ( dv )
                *dv += bu[a] * dbv[b] * p;
        }
    return res;
}

Vector3d BSplineSurface::normal( const Vector2d& uv ) const
{
    Vector3d du, dv;
    (void)eval( uv, &du, &dv );
    return cross( du, dv ).normalized();
}

Vector2d BSplineSurface::project( const Vector3d& p, Vector2d uv, int iterations ) const
{
    const double u0 = knotsU[degreeU], u1 = knotsU[numU()], v0 = knotsV[degreeV], v1 = knotsV[numV()];
    for ( int it = 0; it < iterations; ++it )
    {
        Vector3d su, sv;
        const Vector3d r = p - eval( uv, &su, &sv );
        // Gauss-Newton step: minimize the quadratic model of the squared distance in the domain
        SymMatrix2d h;
        h.xx = dot( su, su );
        h.xy = dot( su, sv );
        h.yy = dot( sv, sv );
        const Vector2d g( dot( su, r ), dot( sv, r ) );
        const auto model = [&]( const Vector2d& x ) { const Vector2d d = x - uv; return dot( d, 0.5 * ( h * d ) - g ); };
        // the minimum in the domain is on the line u = c.x or on the line v = c.y, where c is the minimum in the plane clamped to the domain
        const Vector2d m = uv + h.pseudoinverse() * g;
        const Vector2d c( std::clamp( m.x, u0, u1 ), std::clamp( m.y, v0, v1 ) );
        const Vector2d a( c.x, h.yy > 0 ? std::clamp( uv.y + ( g.y - h.xy * ( c.x - uv.x ) ) / h.yy, v0, v1 ) : c.y );
        const Vector2d b( h.xx > 0 ? std::clamp( uv.x + ( g.x - h.xy * ( c.y - uv.y ) ) / h.xx, u0, u1 ) : c.x, c.y );
        uv = model( a ) <= model( b ) ? a : b;
    }
    return uv;
}

BSplineCurve interpolateBSplineCurve( const std::vector<Vector3d>& points )
{
    const int n = int( points.size() );
    assert( n >= 2 );
    std::vector<double> t( n, 0.0 );
    for ( int i = 1; i < n; ++i )
        t[i] = t[i - 1] + ( points[i] - points[i - 1] ).length();
    for ( auto& x : t )
        x /= std::max( t.back(), 1e-300 );
    t.back() = 1;

    BSplineCurve res;
    res.degree = std::min( 3, n - 1 );
    const int k = res.degree;
    // clamped knots with the inner ones being the averages of the parameters of the points
    res.knots = makeClampedUniformKnots( n, k );
    for ( int j = 1; j < n - k; ++j )
        res.knots[j + k] = std::accumulate( t.begin() + j, t.begin() + j + k, 0.0 ) / k;

    Eigen::MatrixXd m = Eigen::MatrixXd::Zero( n, n );
    Eigen::MatrixXd rhs( n, 3 );
    for ( int i = 0; i < n; ++i )
    {
        double b[cMaxDegree + 1];
        const int i0 = bsplineBasis( res.knots, k, n, t[i], b, nullptr );
        for ( int j = 0; j <= k; ++j )
            m( i, i0 + j ) = b[j];
        for ( int c = 0; c < 3; ++c )
            rhs( i, c ) = points[i][c];
    }
    const Eigen::MatrixXd sol = m.partialPivLu().solve( rhs );
    res.controlPoints.resize( n );
    for ( int i = 0; i < n; ++i )
        res.controlPoints[i] = Vector3d( sol( i, 0 ), sol( i, 1 ), sol( i, 2 ) );
    return res;
}

BSplineSurface fitBSplineSurface( const std::vector<Vector3d>& points, const std::vector<Vector2d>& uvs, const BSplineSurfaceFitSettings& settings )
{
    assert( points.size() == uvs.size() );
    assert( !settings.weights || settings.weights->size() == points.size() );
    const int nu = settings.numU, nv = settings.numV;
    assert( nu >= 4 && nv >= 4 );
    BSplineSurface res;
    res.knotsU = makeClampedUniformKnots( nu, res.degreeU );
    res.knotsV = makeClampedUniformKnots( nv, res.degreeV );
    const int k = nu * nv;
    Eigen::MatrixXd ata = Eigen::MatrixXd::Zero( k, k );
    Eigen::MatrixXd atb = Eigen::MatrixXd::Zero( k, 3 );
    for ( size_t i = 0; i < points.size(); ++i )
    {
        double bu[4], bv[4];
        const int iu = bsplineBasis( res.knotsU, 3, nu, uvs[i].x, bu, nullptr );
        const int iv = bsplineBasis( res.knotsV, 3, nv, uvs[i].y, bv, nullptr );
        const double w = settings.weights ? ( *settings.weights )[i] : 1.0;
        int idx[16];
        double val[16];
        for ( int a = 0; a < 4; ++a )
            for ( int b = 0; b < 4; ++b )
            {
                idx[a * 4 + b] = ( iu + a ) * nv + iv + b;
                val[a * 4 + b] = bu[a] * bv[b];
            }
        for ( int a = 0; a < 16; ++a )
        {
            for ( int b = 0; b < 16; ++b )
                ata( idx[a], idx[b] ) += w * val[a] * val[b];
            for ( int d = 0; d < 3; ++d )
                atb( idx[a], d ) += w * val[a] * points[i][d];
        }
    }
    // smoothing: second differences of the control net
    const int numRows = ( nu - 2 ) * nv + nu * ( nv - 2 );
    const double w2 = settings.smoothing * double( points.size() ) / std::max( numRows, 1 );
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
    res.controlPoints.resize( k );
    for ( int i = 0; i < k; ++i )
        res.controlPoints[i] = Vector3d( sol( i, 0 ), sol( i, 1 ), sol( i, 2 ) );
    return res;
}

} //namespace MR
