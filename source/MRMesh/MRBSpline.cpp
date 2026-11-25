#include "MRBSpline.h"
#include "MRSymMatrix2.h"
#include "MRTimer.h"
#include "MRPch/MREigenSparseCore.h"
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
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
    const int span = int( std::upper_bound( knots.begin() + deg + 1, knots.begin() + n, t ) - knots.begin() ) - 1;
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
    uv = Vector2d( std::clamp( uv.x, u0, u1 ), std::clamp( uv.y, v0, v1 ) );
    Vector3d su, sv;
    Vector3d r = p - eval( uv, &su, &sv );
    for ( int it = 0; it < iterations; ++it )
    {
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
        // the step is halved until the squared distance decreases by at least 1/4 of the decrease -2 * model( next ) predicted
        // by the linearization (a step 2 times too long does not), and the iterations stop when the predicted decrease is below
        // the rounding error of the computed one: the surface point S has the error of a few eps * |S|, and |S| <= |p| + |r|
        const double tol = 8 * std::numeric_limits<double>::epsilon() * r.length() * ( p.length() + r.length() );
        Vector2d next = model( a ) <= model( b ) ? a : b;
        Vector3d nr, nsu, nsv;
        while ( -2 * model( next ) > tol && ( nr = p - eval( next, &nsu, &nsv ) ).lengthSq() > r.lengthSq() + model( next ) / 2 )
            next = 0.5 * ( uv + next );
        if ( !( -2 * model( next ) > tol ) )
            break;
        uv = next;
        r = nr;
        su = nsu;
        sv = nsv;
    }
    return uv;
}

BSplineCurve interpolateBSplineCurve( const std::vector<Vector3d>& points )
{
    const int n = int( points.size() );
    assert( n >= 2 );
    if ( n < 2 )
        return {};
    std::vector<double> t( n, 0.0 );
    for ( int i = 1; i < n; ++i )
    {
        t[i] = t[i - 1] + ( points[i] - points[i - 1] ).length();
        assert( t[i] > t[i - 1] ); // no two consecutive points are equal
    }
    for ( auto& x : t )
        x /= t.back(); // the last one becomes exactly 1

    BSplineCurve res;
    res.degree = std::min( 3, n - 1 );
    const int k = res.degree;
    // clamped knots with the inner ones being the averages of the parameters of the points
    res.knots = makeClampedUniformKnots( n, k );
    for ( int j = 1; j < n - k; ++j )
        res.knots[j + k] = std::accumulate( t.begin() + j, t.begin() + j + k, 0.0 ) / k;

    // the collocation matrix has at most k + 1 non-zeros in a row, and it is not singular for increasing parameters
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve( size_t( n ) * ( k + 1 ) );
    Eigen::MatrixXd rhs( n, 3 );
    for ( int i = 0; i < n; ++i )
    {
        double b[cMaxDegree + 1];
        const int i0 = bsplineBasis( res.knots, k, n, t[i], b, nullptr );
        for ( int j = 0; j <= k; ++j )
            trips.emplace_back( i, i0 + j, b[j] );
        for ( int c = 0; c < 3; ++c )
            rhs( i, c ) = points[i][c];
    }
    Eigen::SparseMatrix<double> m( n, n );
    m.setFromTriplets( trips.begin(), trips.end() );
    // the normal equations
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt( m.transpose() * m );
    assert( ldlt.info() == Eigen::Success );
    res.controlPoints.resize( n, Vector3d::diagonal( std::numeric_limits<double>::quiet_NaN() ) );
    if ( ldlt.info() != Eigen::Success )
        return res; // the preconditions are violated, and solve would leave the solution uninitialized
    const Eigen::MatrixXd sol = ldlt.solve( m.transpose() * rhs );
    for ( int i = 0; i < n; ++i )
        res.controlPoints[i] = Vector3d( sol( i, 0 ), sol( i, 1 ), sol( i, 2 ) );
    return res;
}

BSplineSurface fitBSplineSurface( const std::vector<Vector3d>& points, const std::vector<Vector2d>& uvs, const BSplineSurfaceFitSettings& settings )
{
    MR_TIMER;
    assert( points.size() == uvs.size() );
    assert( !settings.weights || settings.weights->size() == points.size() );
    const int nu = settings.numU, nv = settings.numV;
    assert( nu >= 4 && nv >= 4 );
    if ( nu < 4 || nv < 4 )
        return {};
    BSplineSurface res;
    res.knotsU = makeClampedUniformKnots( nu, res.degreeU );
    res.knotsV = makeClampedUniformKnots( nv, res.degreeV );
    const int k = nu * nv;
    // all points in a knot span have the same 16 non-zero basis functions, so the normal equations are summed per span first,
    // in the upper triangle of the 16 x 16 matrix of each span (control point indices increase with a)
    const int numSpansV = nv - 3;
    std::vector<std::array<double, 256>> spanAta( size_t( nu - 3 ) * numSpansV );
    Eigen::MatrixXd atb = Eigen::MatrixXd::Zero( k, 3 );
    for ( size_t i = 0; i < points.size(); ++i )
    {
        double bu[4], bv[4];
        const int iu = bsplineBasis( res.knotsU, 3, nu, uvs[i].x, bu, nullptr );
        const int iv = bsplineBasis( res.knotsV, 3, nv, uvs[i].y, bv, nullptr );
        const double w = settings.weights ? ( *settings.weights )[i] : 1.0;
        double val[16];
        for ( int a = 0; a < 16; ++a )
            val[a] = bu[a / 4] * bv[a % 4];
        auto& spanA = spanAta[iu * numSpansV + iv];
        for ( int a = 0; a < 16; ++a )
        {
            for ( int b = a; b < 16; ++b )
                spanA[a * 16 + b] += w * val[a] * val[b];
            for ( int d = 0; d < 3; ++d )
                atb( ( iu + a / 4 ) * nv + iv + a % 4, d ) += w * val[a] * points[i][d];
        }
    }
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve( spanAta.size() * 136 + size_t( k ) * 12 );
    for ( int s = 0; s < int( spanAta.size() ); ++s )
    {
        const int i0 = s / numSpansV * nv + s % numSpansV;
        for ( int a = 0; a < 16; ++a )
            for ( int b = a; b < 16; ++b )
                trips.emplace_back( i0 + a / 4 * nv + a % 4, i0 + b / 4 * nv + b % 4, spanAta[s][a * 16 + b] );
    }
    // smoothing: second differences of the control net along u and v divided by the differences of the Greville abscissae
    // ( knots[i + 1] + knots[i + 2] + knots[i + 3] ) / 3, scaled to ( 1, -2, 1 ) for equal differences; they are zero for bilinear functions of ( u, v )
    const int numRows = ( nu - 2 ) * nv + nu * ( nv - 2 );
    const double w2 = settings.smoothing * ( settings.weights ? std::accumulate( settings.weights->begin(), settings.weights->end(), 0.0 ) : double( points.size() ) ) / numRows;
    auto addRow = [&]( const std::vector<double>& knots, int i, int idx1, int step )
    {
        const double d0 = knots[i + 3] - knots[i], d1 = knots[i + 4] - knots[i + 1], h = ( d0 + d1 ) / 2; // 3 times the differences of the abscissae
        const int idx[3] = { idx1 - step, idx1, idx1 + step };
        const double val[3] = { h / d0, -h / d0 - h / d1, h / d1 };
        for ( int a = 0; a < 3; ++a )
            for ( int b = a; b < 3; ++b )
                trips.emplace_back( idx[a], idx[b], w2 * val[a] * val[b] );
    };
    for ( int a = 0; a < nu; ++a )
        for ( int b = 0; b < nv; ++b )
        {
            if ( a > 0 && a + 1 < nu )
                addRow( res.knotsU, a, a * nv + b, nv );
            if ( b > 0 && b + 1 < nv )
                addRow( res.knotsV, b, a * nv + b, 1 );
        }
    Eigen::SparseMatrix<double> ata( k, k );
    ata.setFromTriplets( trips.begin(), trips.end() );
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> ldlt( ata );
    assert( ldlt.info() == Eigen::Success );
    res.controlPoints.resize( k, Vector3d::diagonal( std::numeric_limits<double>::quiet_NaN() ) );
    if ( ldlt.info() != Eigen::Success )
        return res; // the preconditions are violated, and solve would leave the solution uninitialized
    const Eigen::MatrixXd sol = ldlt.solve( atb );
    for ( int i = 0; i < k; ++i )
        res.controlPoints[i] = Vector3d( sol( i, 0 ), sol( i, 1 ), sol( i, 2 ) );
    return res;
}

} //namespace MR
