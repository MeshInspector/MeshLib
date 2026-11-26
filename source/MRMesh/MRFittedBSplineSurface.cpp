#include "MRFittedBSplineSurface.h"
#include "MRBitSet.h"
#include "MRConformalMap.h"
#include "MRMatrix2.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRegionBoundary.h"
#include "MRSymMatrix2.h"
#include "MRTimer.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>

namespace MR
{

namespace
{

/// rotates and scales the parameters into [margin, 1 - margin]^2 with u along their longest direction
void toMarginSquare( Vector<Vector2d>& uv, double margin )
{
    Vector2d c;
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
    Vector2d lo( DBL_MAX, DBL_MAX ), hi( -DBL_MAX, -DBL_MAX );
    for ( auto& p : uv )
    {
        p = Vector2d( dot( p, ev.y ), dot( p, ev.x ) );
        lo = Vector2d( std::min( lo.x, p.x ), std::min( lo.y, p.y ) );
        hi = Vector2d( std::max( hi.x, p.x ), std::max( hi.y, p.y ) );
    }
    for ( auto& p : uv )
    {
        p.x = margin + ( 1 - 2 * margin ) * ( p.x - lo.x ) / std::max( hi.x - lo.x, 1e-30 );
        p.y = margin + ( 1 - 2 * margin ) * ( p.y - lo.y ) / std::max( hi.y - lo.y, 1e-30 );
    }
}

} //anonymous namespace

void FittedBSplineSurface::buildGrid( int size )
{
    assert( size >= 2 && surface.valid() );
    gridSize = std::max( size, 2 );
    grid.resize( size_t( gridSize ) * gridSize );
    for ( int i = 0; i < gridSize; ++i )
        for ( int j = 0; j < gridSize; ++j )
            grid[size_t( i ) * gridSize + j] = surface.eval( gridUV( i, j ) );
}

Vector2d FittedBSplineSurface::gridUV( int i, int j ) const
{
    if ( !surface.valid() || gridSize < 2 )
        return {};
    const double u0 = surface.knotsU[surface.degreeU], u1 = surface.knotsU[surface.numU()];
    const double v0 = surface.knotsV[surface.degreeV], v1 = surface.knotsV[surface.numV()];
    const double a = double( i ) / ( gridSize - 1 ), b = double( j ) / ( gridSize - 1 );
    return Vector2d( u0 * ( 1 - a ) + u1 * a, v0 * ( 1 - b ) + v1 * b );
}

Vector2d FittedBSplineSurface::project( const Vector3d& p ) const
{
    size_t best = 0;
    double bestDistSq = DBL_MAX;
    for ( size_t k = 0; k < grid.size(); ++k )
        if ( const auto d = ( grid[k] - p ).lengthSq(); d < bestDistSq )
        {
            bestDistSq = d;
            best = k;
        }
    return surface.project( p, gridSize > 0 ? gridUV( int( best / gridSize ), int( best % gridSize ) ) : Vector2d() );
}

double FittedBSplineSurface::distanceSq( const Vector3d& p ) const
{
    return ( p - surface.eval( project( p ) ) ).lengthSq();
}

Expected<FittedBSplineSurface> fitBSplineSurfaceToMesh( const MeshPart& mp, const FitBSplineSurfaceToMeshSettings& settings )
{
    MR_TIMER;
    auto map = leastSquaresConformalMap( mp );
    if ( !map )
        return unexpected( std::move( map.error() ) );
    const auto& topology = mp.mesh.topology;
    const auto& faces = topology.getFaceIds( mp.region );
    const auto verts = getIncidentVerts( topology, faces );
    const auto num = makeVectorWithSeqNums( verts );
    Vector<Vector3d> pts;
    Vector<Vector2d> uvs;
    for ( auto v : verts )
    {
        pts.push_back( Vector3d( mp.mesh.points[v] ) );
        uvs.push_back( Vector2d( ( *map )[v] ) );
        if ( !std::isfinite( uvs.back().x + uvs.back().y ) )
            return unexpected( "Not finite parameters" );
    }
    toMarginSquare( uvs, settings.margin );
    const size_t numVerts = pts.size();

    // points inside the triangles much larger than the typical one: without them the surface is free to bulge there
    Vector<double> lens;
    lens.reserve( faces.count() * 3 );
    for ( auto f : faces )
    {
        const auto t = topology.getTriVerts( f );
        for ( int i = 0; i < 3; ++i )
            lens.push_back( ( pts[num[t[i]]] - pts[num[t[( i + 1 ) % 3]]] ).length() );
    }
    const auto mid = lens.begin() + lens.size() / 2;
    std::nth_element( lens.begin(), mid, lens.end() );
    const double med = *mid;
    for ( auto f : faces )
    {
        const auto t = topology.getTriVerts( f );
        const Vector3d p[3] = { pts[num[t[0]]], pts[num[t[1]]], pts[num[t[2]]] };
        const Vector2d q[3] = { uvs[num[t[0]]], uvs[num[t[1]]], uvs[num[t[2]]] };
        double longest = 0;
        for ( int i = 0; i < 3; ++i )
            longest = std::max( longest, ( p[i] - p[( i + 1 ) % 3] ).length() );
        if ( longest <= 3 * med )
            continue;
        const int n = int( std::min( 8.0, std::ceil( longest / ( 1.5 * med ) ) ) );
        for ( int i = 0; i <= n; ++i )
            for ( int j = 0; i + j <= n; ++j )
            {
                const int k = n - i - j;
                if ( std::max( { i, j, k } ) == n )
                    continue; // a vertex
                const double b0 = double( i ) / n, b1 = double( j ) / n, b2 = double( k ) / n;
                pts.push_back( b0 * p[0] + b1 * p[1] + b2 * p[2] );
                uvs.push_back( b0 * q[0] + b1 * q[1] + b2 * q[2] );
            }
    }
    Vector<double> weights( pts.size(), 0.01 );
    std::fill( weights.begin(), weights.begin() + numVerts, 1.0 );

    std::optional<FittedBSplineSurface> best;
    static constexpr int sizes[][2] = { { 6, 4 }, { 8, 4 }, { 10, 5 }, { 14, 5 }, { 18, 6 }, { 24, 6 }, { 32, 7 }, { 40, 8 } };
    for ( const auto& sz : sizes )
    {
        if ( sz[0] * sz[1] > 0.7 * numVerts )
            break;
        const BSplineSurfaceFitSettings fitSettings{ .numU = sz[0], .numV = sz[1], .smoothing = settings.smoothing, .weights = &weights };
        auto uv = uvs;
        std::optional<BSplineSurface> fit;
        for ( int it = 0; it < 8; ++it )
        {
            fit = fitBSplineSurface( pts, uv, fitSettings );
            if ( !fit )
                break;
            for ( size_t i = 0; i < pts.size(); ++i )
                uv[i] = fit->project( pts[i], uv[i], 4 );
        }
        if ( !fit )
            continue;
        FittedBSplineSurface s;
        s.surface = std::move( *fit );
        s.buildGrid();
        // projection without initial parameters also catches folds of the surface
        double maxDistSq = 0;
        for ( size_t i = 0; i < numVerts; ++i )
            maxDistSq = std::max( maxDistSq, s.distanceSq( pts[i] ) );
        s.maxError = std::sqrt( maxDistSq );
        if ( !std::isfinite( s.maxError ) )
            continue;
        if ( !best || s.maxError < best->maxError )
            best = std::move( s );
        if ( best->maxError < settings.tolerance )
            break;
    }
    if ( !best )
        return unexpected( numVerts < 35 ? "Too few vertices" : "Surface fit failed" );
    return std::move( *best );
}

} //namespace MR
