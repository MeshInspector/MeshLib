#include "MRFittedBSplineSurface.h"
#include "MRBitSet.h"
#include "MRBox.h"
#include "MRConstants.h"
#include "MRConformalMap.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>

namespace MR
{

namespace
{

/// rotates the parameters to their bounding rectangle of the smallest area (among the rotations by whole degrees) with u along its longer side,
/// and scales them into [margin, 1 - margin]^2; it is not a reflection, so the orientation of the triangles is kept
void toMarginSquare( Vector<Vector2d>& uv, double margin )
{
    const auto rotated = []( const Vector2d& p, const Vector2d& d ) { return Vector2d( dot( p, d ), cross( d, p ) ); };
    const auto findBox = [&]( const Vector2d& d )
    {
        Box2d box;
        for ( const auto& p : uv )
            box.include( rotated( p, d ) );
        return box;
    };
    Vector2d best( 1, 0 );
    double bestArea = DBL_MAX;
    for ( int deg = 0; deg < 90; ++deg )
    {
        const double a = deg * PI / 180;
        const Vector2d d( std::cos( a ), std::sin( a ) );
        const auto size = findBox( d ).size();
        if ( const double area = size.x * size.y; area < bestArea )
        {
            bestArea = area;
            best = size.x >= size.y ? d : Vector2d( -d.y, d.x );
        }
    }
    const auto box = findBox( best );
    const auto size = box.size();
    for ( auto& p : uv )
    {
        const auto q = rotated( p, best ) - box.min;
        p.x = margin + ( 1 - 2 * margin ) * q.x / std::max( size.x, 1e-30 );
        p.y = margin + ( 1 - 2 * margin ) * q.y / std::max( size.y, 1e-30 );
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
    assert( !grid.empty() );
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
    if ( !( settings.margin >= 0 && settings.margin < 0.5 ) )
        return unexpected( "Wrong margin" );
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
    Vector<double> distSq( numVerts );
    static constexpr int sizes[][2] = { { 6, 4 }, { 8, 4 }, { 10, 5 }, { 14, 5 }, { 18, 6 }, { 24, 6 }, { 32, 7 }, { 40, 8 } };
    for ( const auto& sz : sizes )
    {
        if ( sz[0] * sz[1] > 0.7 * numVerts )
            break;
        const BSplineSurfaceFitSettings fitSettings{ .numU = sz[0], .numV = sz[1], .smoothing = settings.smoothing, .weights = &weights };
        auto uv = uvs;
        std::optional<BSplineSurface> fit;
        const int numFits = 8;
        for ( int it = 0; it < numFits; ++it )
        {
            fit = fitBSplineSurface( pts, uv, fitSettings );
            if ( !fit || it + 1 == numFits )
                break;
            ParallelFor( size_t( 0 ), pts.size(), [&]( size_t i )
            {
                uv[i] = fit->project( pts[i], uv[i], 4 );
            } );
        }
        if ( !fit )
            continue;
        FittedBSplineSurface s;
        s.surface = std::move( *fit );
        s.buildGrid();
        // projection without initial parameters also catches folds of the surface
        ParallelFor( size_t( 0 ), numVerts, [&]( size_t i )
        {
            distSq[i] = s.distanceSq( pts[i] );
        } );
        double maxDistSq = 0;
        for ( double d : distSq )
            maxDistSq = std::isfinite( d ) ? std::max( maxDistSq, d ) : DBL_MAX;
        if ( maxDistSq == DBL_MAX )
            continue;
        s.maxError = std::sqrt( maxDistSq );
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
