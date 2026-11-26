#include "MRMeshToStepBSpline.hpp"
#include "MRBitSet.h"
#include "MRConformalMap.h"
#include "MRMatrix2.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRegionBoundary.h"
#include "MRSymMatrix2.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace MR
{

namespace MeshToStep
{

namespace
{

/// the number of grid points along u and v
constexpr int cGrid = 60;

/// the part of the surface domain occupied by fitted points is [cMargin, 1-cMargin]^2, the rest is a natural extension
constexpr double cMargin = 0.06;

V2 gridUV( int k )
{
    return V2( double( k / cGrid ) / ( cGrid - 1 ), double( k % cGrid ) / ( cGrid - 1 ) );
}

/// rotates and scales the parameters into [cMargin, 1-cMargin]^2 with u along their longest direction
void toMarginSquare( Vector<V2>& uv )
{
    V2 c;
    for ( const auto& p : uv )
        c += p / double( uv.size() );
    SymMatrix2d cov;
    for ( auto& p : uv )
    {
        p -= c;
        cov += outerSquare( p );
    }
    Matrix2d ev;
    cov.eigens( &ev );
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
}

} //anonymous namespace

void BSplineSurf::buildGrid()
{
    grid.resize( cGrid * cGrid );
    for ( int k = 0; k < cGrid * cGrid; ++k )
        grid[k] = surf.eval( gridUV( k ) );
}

V2 BSplineSurf::project( const V3& p ) const
{
    int best = 0;
    double bestD = DBL_MAX;
    for ( int k = 0; k < (int)grid.size(); ++k )
        if ( auto d = ( grid[k] - p ).lengthSq(); d < bestD )
        {
            bestD = d;
            best = k;
        }
    return surf.project( p, gridUV( best ) );
}

V3 BSplineSurf::normalAt( const V2& uv ) const
{
    const V3 n = surf.normal( uv );
    return n == V3() ? V3( 0, 0, 1 ) : n;
}

std::shared_ptr<BSplineSurf> fitBSpline( const Mesh& mesh, const FaceBitSet& region, double tol )
{
    const auto map = leastSquaresConformalMap( { mesh, &region } );
    if ( !map )
        return {};
    const auto verts = getIncidentVerts( mesh.topology, region );
    const auto num = makeVectorWithSeqNums( verts );
    Vector<V3> allPts;
    Vector<V2> allUV;
    for ( auto v : verts )
    {
        allPts.push_back( V3( mesh.points[v] ) );
        allUV.push_back( V2( ( *map )[v] ) );
        if ( !std::isfinite( allUV.back().x + allUV.back().y ) )
            return {};
    }
    toMarginSquare( allUV );
    const size_t numVerts = allPts.size();

    // points inside the triangles much larger than the typical one: without them the surface is free to bulge there
    Vector<double> lens;
    lens.reserve( region.count() * 3 );
    for ( auto f : region )
    {
        const auto t = mesh.topology.getTriVerts( f );
        for ( int i = 0; i < 3; ++i )
            lens.push_back( ( allPts[num[t[i]]] - allPts[num[t[( i + 1 ) % 3]]] ).length() );
    }
    if ( lens.empty() )
        return {};
    auto mid = lens.begin() + lens.size() / 2;
    std::nth_element( lens.begin(), mid, lens.end() );
    const double med = *mid;
    for ( auto f : region )
    {
        const auto t = mesh.topology.getTriVerts( f );
        const V3 p[3] = { allPts[num[t[0]]], allPts[num[t[1]]], allPts[num[t[2]]] };
        const V2 q[3] = { allUV[num[t[0]]], allUV[num[t[1]]], allUV[num[t[2]]] };
        double longest = 0;
        for ( int i = 0; i < 3; ++i )
            longest = std::max( longest, ( p[i] - p[( i + 1 ) % 3] ).length() );
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
                allPts.push_back( b0 * p[0] + b1 * p[1] + b2 * p[2] );
                allUV.push_back( b0 * q[0] + b1 * q[1] + b2 * q[2] );
            }
    }
    Vector<double> weights( allPts.size(), 0.01 );
    std::fill( weights.begin(), weights.begin() + numVerts, 1.0 );

    std::shared_ptr<BSplineSurf> best;
    static constexpr int sizes[][2] = { { 6, 4 }, { 8, 4 }, { 10, 5 }, { 14, 5 }, { 18, 6 }, { 24, 6 }, { 32, 7 }, { 40, 8 } };
    for ( const auto& sz : sizes )
    {
        if ( sz[0] * sz[1] > 0.7 * numVerts )
            break;
        const BSplineSurfaceFitSettings settings{ .numU = sz[0], .numV = sz[1], .smoothing = 1e-7, .weights = &weights };
        auto uv = allUV;
        std::optional<BSplineSurface> fit;
        for ( int it = 0; it < 8; ++it )
        {
            fit = fitBSplineSurface( allPts, uv, settings );
            if ( !fit )
                break;
            for ( size_t i = 0; i < allPts.size(); ++i )
                uv[i] = fit->project( allPts[i], uv[i], 4 );
        }
        if ( !fit )
            continue;
        auto s = std::make_shared<BSplineSurf>();
        s->surf = std::move( *fit );
        s->buildGrid();
        // projection from scratch also catches folds of the surface
        for ( size_t i = 0; i < numVerts; ++i )
            s->err = std::max( s->err, s->absDist( allPts[i] ) );
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
