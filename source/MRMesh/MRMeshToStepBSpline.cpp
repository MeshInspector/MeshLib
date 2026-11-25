#include "MRMeshToStepBSpline.hpp"
#include "MRMeshToStepSurf.hpp"
#include "MRMatrix2.h"
#include "MRMatrix3.h"
#include "MRSymMatrix2.h"
#include "MRSymMatrix3.h"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

namespace MR
{

namespace MeshToStep
{

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

namespace
{

/// the part of the surface domain occupied by fitted points is [cMargin, 1-cMargin]^2, the rest is a natural extension
constexpr double cMargin = 0.06;

/// least squares conformal map with free boundary, rotated and scaled into [cMargin, 1-cMargin]^2 with u along the longest direction
std::vector<V2> lscm( const std::vector<V3>& pts, const std::vector<Tri>& tris )
{
    const int n = int( pts.size() );
    // pin two vertices most distant along the main direction
    const V3 mean = centroid( pts );
    SymMatrix3d cov;
    for ( const auto& p : pts )
        cov += outerSquare( p - mean );
    Matrix3d vecs;
    cov.eigens( &vecs );
    const V3 axis = vecs.z;
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
    SymMatrix2d cov2;
    for ( auto& p : uv )
    {
        p -= c;
        cov2 += outerSquare( p );
    }
    Matrix2d ev;
    cov2.eigens( &ev );
    V2 lo( DBL_MAX, DBL_MAX ), hi( -DBL_MAX, -DBL_MAX );
    for ( auto& p : uv )
    {
        p = V2( dot( p, ev.y ), dot( p, ev.x ) );
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

} //anonymous namespace

std::shared_ptr<BSplineSurf> fitBSpline( const std::vector<V3>& pts, const std::vector<Tri>& tris, double tol )
{
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

} //namespace MeshToStep

} //namespace MR
