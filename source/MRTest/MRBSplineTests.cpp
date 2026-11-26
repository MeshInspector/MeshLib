#include <MRMesh/MRBSpline.h>
#include <MRMesh/MRConformalMap.h>
#include <MRMesh/MRFittedBSplineSurface.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshPart.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRRegularGridMesh.h>
#include <MRMesh/MRVector.h>
#include <gtest/gtest.h>
#include <limits>

namespace MR
{

TEST( MRMesh, InterpolateBSplineCurve )
{
    // a helix
    Vector<Vector3d> pts;
    for ( int i = 0; i < 40; ++i )
        pts.emplace_back( std::cos( 0.2 * i ), std::sin( 0.2 * i ), 0.05 * i );
    const auto c = interpolateBSplineCurve( pts ).value();
    EXPECT_EQ( c.degree, 3 );
    ASSERT_EQ( c.controlPoints.size(), pts.size() );
    ASSERT_EQ( c.knots.size(), pts.size() + 4 );
    // the parameters are proportional to the length of the polyline, which is the same between all neighbours
    for ( int i = 0; i < 40; ++i )
        EXPECT_NEAR( ( c.eval( i / 39.0 ) - pts[i] ).length(), 0, 1e-12 );
    // between the points the curve is close to the helix
    for ( int i = 0; i < 39; ++i )
    {
        const auto p = c.eval( ( i + 0.5 ) / 39.0 );
        EXPECT_NEAR( Vector2d( p.x, p.y ).length(), 1, 1e-4 );
    }

    EXPECT_FALSE( interpolateBSplineCurve( { Vector3d() } ).has_value() );
    // equal consecutive points, all points equal, not finite coordinates
    EXPECT_FALSE( interpolateBSplineCurve( { Vector3d(), Vector3d( 1, 0, 0 ), Vector3d( 1, 0, 0 ), Vector3d( 2, 0, 0 ) } ).has_value() );
    EXPECT_FALSE( interpolateBSplineCurve( { Vector3d( 1, 2, 3 ), Vector3d( 1, 2, 3 ) } ).has_value() );
    EXPECT_FALSE( interpolateBSplineCurve( { Vector3d(), Vector3d( std::numeric_limits<double>::infinity(), 0, 0 ) } ).has_value() );
    EXPECT_FALSE( interpolateBSplineCurve( { Vector3d(), Vector3d( std::numeric_limits<double>::quiet_NaN(), 0, 0 ) } ).has_value() );
}

TEST( MRMesh, BSplineNotValid )
{
    BSplineCurve curve;
    EXPECT_FALSE( curve.valid() );
    curve.knots = { 0, 0, 0, 0, 1, 1, 1, 1 };
    curve.controlPoints = { Vector3d( 1, 0, 0 ), Vector3d( 2, 0, 0 ), Vector3d( 3, 0, 0 ), Vector3d( 4, 0, 0 ) };
    EXPECT_TRUE( curve.valid() );
    EXPECT_EQ( curve.eval( 1 ), Vector3d( 4, 0, 0 ) );
    // too large degree, or a wrong number of knots
    curve.degree = 8;
    EXPECT_FALSE( curve.valid() );
    EXPECT_EQ( curve.eval( 1 ), Vector3d() );
    curve.degree = 3;
    curve.knots.push_back( 1 );
    EXPECT_FALSE( curve.valid() );
    EXPECT_EQ( curve.eval( 1 ), Vector3d() );

    BSplineSurface surf;
    EXPECT_FALSE( surf.valid() );
    surf.knotsU = surf.knotsV = { 0, 0, 0, 0, 1, 1, 1, 1 };
    surf.controlPoints.resize( 16, Vector3d( 1, 2, 3 ) );
    EXPECT_TRUE( surf.valid() );
    EXPECT_NEAR( ( surf.eval( { 0.3, 0.6 } ) - Vector3d( 1, 2, 3 ) ).length(), 0, 1e-12 );
    surf.controlPoints.pop_back();
    EXPECT_FALSE( surf.valid() );
    Vector3d du( 1, 1, 1 ), dv( 1, 1, 1 );
    EXPECT_EQ( surf.eval( { 0.3, 0.6 }, &du, &dv ), Vector3d() );
    EXPECT_EQ( du, Vector3d() );
    EXPECT_EQ( dv, Vector3d() );
    EXPECT_EQ( surf.normal( { 0.3, 0.6 } ), Vector3d() );
    EXPECT_EQ( surf.project( { 5, 5, 5 }, { 0.3, 0.6 } ), Vector2d( 0.3, 0.6 ) );
    surf.controlPoints.resize( 16 );
    surf.degreeV = 9;
    EXPECT_FALSE( surf.valid() );
    EXPECT_EQ( surf.eval( { 0.3, 0.6 } ), Vector3d() );

    const auto line = interpolateBSplineCurve( { Vector3d( 0, 0, 0 ), Vector3d( 2, 0, 0 ) } ).value();
    EXPECT_EQ( line.degree, 1 );
    EXPECT_NEAR( ( line.eval( 0.25 ) - Vector3d( 0.5, 0, 0 ) ).length(), 0, 1e-15 );

    const Vector<Vector3d> three = { Vector3d( 0, 0, 0 ), Vector3d( 1, 1, 0 ), Vector3d( 2, 0, 0 ) };
    const auto parabola = interpolateBSplineCurve( three ).value();
    EXPECT_EQ( parabola.degree, 2 );
    for ( int i = 0; i < 3; ++i )
        EXPECT_NEAR( ( parabola.eval( i / 2.0 ) - three[i] ).length(), 0, 1e-15 );

    // 4 points with equal distances: a cubic without inner knots
    const Vector<Vector3d> four = { Vector3d( 0, 0, 0 ), Vector3d( 1, 1, 0 ), Vector3d( 2, 1, 1 ), Vector3d( 3, 0, 1 ) };
    const auto cubic = interpolateBSplineCurve( four ).value();
    EXPECT_EQ( cubic.knots, Vector<double>( { 0, 0, 0, 0, 1, 1, 1, 1 } ) );
    for ( int i = 0; i < 4; ++i )
        EXPECT_NEAR( ( cubic.eval( i / 3.0 ) - four[i] ).length(), 0, 1e-12 );

    // uneven spacing: the parameters are proportional to the length of the polyline;
    // the second chain has two steps of 1e-5, where the normal equations of the collocation system were 5e-6 off
    for ( const auto& uneven : { Vector<Vector3d>{ Vector3d( 0, 0, 0 ), Vector3d( 0.01, 0, 0 ), Vector3d( 1, 1, 0 ), Vector3d( 1, 1, 2 ), Vector3d( 5, 1, 2 ) },
        Vector<Vector3d>{ Vector3d( 0, 0, 0 ), Vector3d( 1, 0, 0 ), Vector3d( 1, 1e-5, 0 ), Vector3d( 1, 1e-5, 1e-5 ), Vector3d( 2, 1, 1 ), Vector3d( 3, 0, 1 ) } } )
    {
        const auto uc = interpolateBSplineCurve( uneven ).value();
        Vector<double> t( 1, 0.0 );
        for ( size_t i = 1; i < uneven.size(); ++i )
            t.push_back( t.back() + ( uneven[i] - uneven[i - 1] ).length() );
        for ( size_t i = 0; i < uneven.size(); ++i )
            EXPECT_NEAR( ( uc.eval( t[i] / t.back() ) - uneven[i] ).length(), 0, 1e-9 );
    }
}

TEST( MRMesh, FitBSplineSurface )
{
    Vector<Vector3d> pts;
    Vector<Vector2d> uvs;
    for ( int i = 0; i <= 20; ++i )
        for ( int j = 0; j <= 20; ++j )
        {
            const double u = i / 20.0, v = j / 20.0;
            uvs.emplace_back( u, v );
            pts.emplace_back( 3 * u, 2 * v, 2 * u - v + 1 );
        }
    EXPECT_FALSE( fitBSplineSurface( pts, uvs, { .numU = 3 } ).has_value() );
    EXPECT_FALSE( fitBSplineSurface( pts, uvs, { .numV = 3 } ).has_value() );
    EXPECT_FALSE( fitBSplineSurface( {}, {} ).has_value() ); // the zero matrix fails the factorization
    // B-splines reproduce linear functions exactly
    auto s = fitBSplineSurface( pts, uvs ).value();
    EXPECT_EQ( s.numU(), 4 );
    EXPECT_EQ( s.numV(), 4 );
    for ( size_t i = 0; i < pts.size(); ++i )
        EXPECT_NEAR( ( s.eval( uvs[i] ) - pts[i] ).length(), 0, 1e-9 );

    // an outlier with zero weight does not change the surface
    {
        auto wPts = pts;
        auto wUvs = uvs;
        wPts.emplace_back( 1.5, 1, 100 );
        wUvs.emplace_back( 0.5, 0.5 );
        Vector<double> weights( wPts.size(), 1.0 );
        weights.back() = 0;
        const auto ws = fitBSplineSurface( wPts, wUvs, { .weights = &weights } ).value();
        EXPECT_NEAR( ( ws.eval( wUvs.back() ) - Vector3d( 1.5, 1, 1.5 ) ).length(), 0, 1e-9 );
    }

    // the weights multiply the squared distances: the plane shifted by -3 with weight 1 and by +1 with weight 3 give the plane
    {
        auto wPts = pts;
        auto wUvs = uvs;
        Vector<double> weights( 2 * pts.size(), 1.0 );
        for ( size_t i = 0; i < pts.size(); ++i )
        {
            wPts[i].z -= 3;
            wPts.push_back( pts[i] + Vector3d( 0, 0, 1 ) );
            wUvs.push_back( uvs[i] );
            weights[pts.size() + i] = 3;
        }
        const auto ws = fitBSplineSurface( wPts, wUvs, { .weights = &weights } ).value();
        for ( size_t i = 0; i < pts.size(); ++i )
            EXPECT_NEAR( ( ws.eval( uvs[i] ) - pts[i] ).length(), 0, 1e-9 );
    }

    // the smoothing keeps bilinear functions of (u, v): such a surface with points only in an ellipse is reproduced in all the domain
    const auto bilinear = []( const Vector2d& uv ) { return Vector3d( 3 * uv.x, 2 * uv.y, 2 * uv.x - uv.y + uv.x * uv.y ); };
    Vector<Vector3d> ePts;
    Vector<Vector2d> eUvs;
    for ( const auto& uv : uvs )
    {
        if ( sqr( ( uv.x - 0.5 ) / 0.44 ) + sqr( ( uv.y - 0.5 ) / 0.3 ) > 1 )
            continue;
        eUvs.push_back( uv );
        ePts.push_back( bilinear( uv ) );
    }
    for ( const auto& [nu, nv] : { std::pair( 6, 4 ), std::pair( 10, 5 ) } )
    {
        const auto es = fitBSplineSurface( ePts, eUvs, { .numU = nu, .numV = nv } ).value();
        EXPECT_EQ( es.numU(), nu );
        EXPECT_EQ( es.numV(), nv );
        for ( const auto& uv : { Vector2d( 0, 0 ), Vector2d( 1, 0 ), Vector2d( 0, 1 ), Vector2d( 1, 1 ), Vector2d( 0.3, 0.6 ) } )
            EXPECT_NEAR( ( es.eval( uv ) - bilinear( uv ) ).length(), 0, 1e-8 );
    }

    // a curved surface
    for ( size_t i = 0; i < pts.size(); ++i )
        pts[i].z = std::sin( 3 * uvs[i].x ) * std::cos( 2 * uvs[i].y );
    s = fitBSplineSurface( pts, uvs, { .numU = 8, .numV = 8 } ).value();
    for ( size_t i = 0; i < pts.size(); ++i )
        EXPECT_NEAR( ( s.eval( uvs[i] ) - pts[i] ).length(), 0, 1e-3 );

    // projection of a point above the surface
    const Vector2d uv0( 0.3, 0.7 );
    const auto p = s.eval( uv0 ) + 0.01 * s.normal( uv0 );
    const auto uv = s.project( p, Vector2d( 0.5, 0.5 ) );
    EXPECT_NEAR( ( uv - uv0 ).length(), 0, 1e-7 ); // steps below ~1e-9 do not change the computed distance
}

TEST( MRMesh, ProjectOnBSplineSurface )
{
    BSplineSurface s;
    s.degreeU = s.degreeV = 1;
    s.knotsU = s.knotsV = { 0, 0, 1, 1 };
    // the plane u * (1,0,0) + v * (1,1,0) with not orthogonal derivatives, the closest points are on the border of the domain
    s.controlPoints = { Vector3d( 0, 0, 0 ), Vector3d( 1, 1, 0 ), Vector3d( 1, 0, 0 ), Vector3d( 2, 1, 0 ) };
    EXPECT_NEAR( ( s.project( Vector3d( 0, 0.8, 0 ), Vector2d( 0.5, 0.5 ) ) - Vector2d( 0, 0.4 ) ).length(), 0, 1e-12 );
    EXPECT_NEAR( ( s.project( Vector3d( 0.5, -2, 0 ), Vector2d( 0.5, 0.5 ) ) - Vector2d( 0.5, 0 ) ).length(), 0, 1e-12 );
    EXPECT_NEAR( ( s.project( Vector3d( 2.1, 0.5, 0 ), Vector2d( 2, 0.5 ) ) - Vector2d( 1, 0.8 ) ).length(), 0, 1e-12 ); // from outside the domain

    // a collapsed row of control points: the derivative along v is zero at u = 0
    s.controlPoints = { Vector3d( 0, 0, 0 ), Vector3d( 0, 0, 0 ), Vector3d( 1, 0, 0 ), Vector3d( 1, 1, 0 ) };
    EXPECT_NEAR( ( s.project( Vector3d( 0.5, 0.25, 0 ), Vector2d( 0, 0.9 ) ) - Vector2d( 0.5, 0.5 ) ).length(), 0, 1e-12 );

    // the arch z = (1 - x^2) / 2 of curvature radius 1 at the apex, and points at distances 3 and 25 above it,
    // where the Gauss-Newton steps are 4 and 26 times too long
    s.degreeU = 2;
    s.knotsU = { 0, 0, 0, 1, 1, 1 };
    s.controlPoints = { Vector3d( -1, 0, 0 ), Vector3d( -1, 1, 0 ), Vector3d( 0, 0, 1 ), Vector3d( 0, 1, 1 ), Vector3d( 1, 0, 0 ), Vector3d( 1, 1, 0 ) };
    EXPECT_NEAR( ( s.project( Vector3d( 0, 0.5, 3.5 ), Vector2d( 0.9, 0.2 ) ) - Vector2d( 0.5, 0.5 ) ).length(), 0, 1e-6 );
    EXPECT_NEAR( ( s.project( Vector3d( 0, 0.5, 25.5 ), Vector2d( 0.9, 0.2 ) ) - Vector2d( 0.5, 0.5 ) ).length(), 0, 1e-6 );
}

TEST( MRMesh, LeastSquaresConformalMap )
{
    // a grid on an inclined plane, the conformal map of its part with x <= 5 is a similarity
    const int nx = 8, ny = 4;
    const auto mesh = makeRegularGridMesh( nx, ny, []( size_t, size_t ) { return true; },
        []( size_t x, size_t y ) { return Vector3f( float( x ), 0.7f * y, 0.3f * x + 0.2f * y ); } ).value();
    VertBitSet left( nx * ny );
    for ( int v = 0; v < nx * ny; ++v )
        left.set( VertId( v ), v % nx <= 5 );
    const auto region = getInnerFaces( mesh.topology, left );
    const auto uv = leastSquaresConformalMap( { mesh, &region } );
    ASSERT_TRUE( uv.has_value() ) << uv.error();
    const VertId v0( 0 ), v1( nx * ny - 3 );
    const double scale = ( ( *uv )[v1] - ( *uv )[v0] ).length() / ( mesh.points[v1] - mesh.points[v0] ).length();
    for ( auto a : mesh.topology.getValidVerts() )
    {
        if ( int( a ) % nx > 5 )
        {
            EXPECT_EQ( ( *uv )[a], UVCoord() ); // outside the region
            continue;
        }
        for ( auto b : mesh.topology.getValidVerts() )
        {
            if ( int( b ) % nx > 5 )
                continue;
            EXPECT_NEAR( ( ( *uv )[a] - ( *uv )[b] ).length(), scale * ( mesh.points[a] - mesh.points[b] ).length(), 1e-5 );
        }
    }
    // not mirrored: the triangles are counterclockwise seen from the normal of the plane, and in the map
    for ( auto f : region )
    {
        const auto tv = mesh.topology.getTriVerts( f );
        EXPECT_GT( cross( ( *uv )[tv[1]] - ( *uv )[tv[0]], ( *uv )[tv[2]] - ( *uv )[tv[0]] ), 0 );
    }

    // a fan around a raised vertex, not developable (312 degrees around it): the reference coordinates minimize the Dirichlet energy minus
    // the area of the map with the extreme vertices along x pinned; unit weights of the triangles in the least squares are 7e-3 off
    const auto fan = Mesh::fromTriangles( { Vector3f( 0, 0, 1.5f ), Vector3f( 3, 0, 0 ), Vector3f( 1, 2, 0.5f ), Vector3f( -1, 1.5f, -0.5f ),
        Vector3f( -3, 0, 0.25f ), Vector3f( -1, -2, 0 ), Vector3f( 1.5f, -1.5f, -0.25f ) },
        { { 0_v, 1_v, 2_v }, { 0_v, 2_v, 3_v }, { 0_v, 3_v, 4_v }, { 0_v, 4_v, 5_v }, { 0_v, 5_v, 6_v }, { 0_v, 6_v, 1_v } } );
    const UVCoord fanRef[] = { { 0.491746f, 0.024721f }, { 1, 0 }, { 0.661545f, 0.346254f }, { 0.263965f, 0.353664f },
        { 0, 0 }, { 0.333594f, -0.343738f }, { 0.767868f, -0.279723f } };
    const auto fanUv = leastSquaresConformalMap( fan );
    ASSERT_TRUE( fanUv.has_value() ) << fanUv.error();
    for ( int v = 0; v < 7; ++v )
        EXPECT_NEAR( ( ( *fanUv )[VertId( v )] - fanRef[v] ).length(), 0, 1e-5 );

    // vertex 3 is only in a degenerate triangle, so its coordinates are not defined, and the degenerate triangle alone gives the zero matrix
    VertCoords degPoints{ Vector3f( 0, 0, 0 ), Vector3f( 2, 0, 0 ), Vector3f( 1, 1, 0 ), Vector3f( 1, 0, 0 ) };
    Triangulation degTris{ { VertId( 0 ), VertId( 1 ), VertId( 2 ) }, { VertId( 1 ), VertId( 0 ), VertId( 3 ) } };
    const auto degMesh = Mesh::fromTriangles( std::move( degPoints ), degTris );
    EXPECT_FALSE( leastSquaresConformalMap( degMesh ).has_value() );
    FaceBitSet degRegion( 2 );
    degRegion.set( FaceId( 1 ) );
    EXPECT_FALSE( leastSquaresConformalMap( { degMesh, &degRegion } ).has_value() );

    auto nanMesh = mesh;
    nanMesh.points[v0].z = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE( leastSquaresConformalMap( nanMesh ).has_value() );
}

TEST( MRMesh, FitBSplineSurfaceToMesh )
{
    // a curved patch 3 x 2 with the grid step 0.1
    const auto surfPoint = []( double x, double y ) { return Vector3d( x, y, 0.3 * std::sin( x ) * std::cos( 1.5 * y ) ); };
    const auto mesh = makeRegularGridMesh( 31, 21, []( size_t, size_t ) { return true; },
        [&]( size_t x, size_t y ) { return Vector3f( surfPoint( 0.1 * x, 0.1 * y ) ); } ).value();
    const auto s = fitBSplineSurfaceToMesh( mesh, { .tolerance = 1e-3 } ).value();
    EXPECT_TRUE( s.surface.valid() );
    EXPECT_LT( s.maxError, 1e-3 );
    EXPECT_EQ( s.gridSize, 60 );
    ASSERT_EQ( s.grid.size(), 3600 );
    EXPECT_EQ( s.gridUV( 59, 0 ), Vector2d( 1, 0 ) );
    EXPECT_EQ( s.grid[59 * 60], s.surface.eval( Vector2d( 1, 0 ) ) );
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const Vector3d p( mesh.points[v] );
        EXPECT_LE( std::sqrt( s.distanceSq( p ) ), s.maxError );
        // the vertices are inside the margin of the domain
        const auto uv = s.project( p );
        EXPECT_GT( std::min( uv.x, uv.y ), 0.05 );
        EXPECT_LT( std::max( uv.x, uv.y ), 0.95 );
    }
    // the normals of the surface and of the mesh agree
    for ( auto f : mesh.topology.getValidFaces() )
        EXPECT_GT( dot( s.surface.normal( s.project( Vector3d( mesh.triCenter( f ) ) ) ), Vector3d( mesh.normal( f ) ) ), 0.99 );
    // u is along the longest direction of the patch
    EXPECT_GT( ( s.surface.eval( Vector2d( 1, 0.5 ) ) - s.surface.eval( Vector2d( 0, 0.5 ) ) ).length(), 3 );
    EXPECT_LT( ( s.surface.eval( Vector2d( 0.5, 1 ) ) - s.surface.eval( Vector2d( 0.5, 0 ) ) ).length(), 3 );
    // the surface is close to the patch between the vertices too
    EXPECT_LT( s.distanceSq( surfPoint( 1.234, 0.777 ) ), 4e-6 ); // 2e-3 squared
    // projection without initial parameters of a point above the surface
    const Vector2d uv0( 0.4, 0.6 );
    const auto above = s.surface.eval( uv0 ) + 0.05 * s.surface.normal( uv0 );
    EXPECT_NEAR( ( s.project( above ) - uv0 ).length(), 0, 1e-6 );
    EXPECT_NEAR( std::sqrt( s.distanceSq( above ) ), 0.05, 1e-9 );

    // a coarser grid
    auto coarse = s;
    coarse.buildGrid( 5 );
    EXPECT_EQ( coarse.grid.size(), 25 );
    EXPECT_EQ( coarse.gridUV( 2, 4 ), Vector2d( 0.5, 1 ) );
    EXPECT_NEAR( ( coarse.project( above ) - uv0 ).length(), 0, 1e-6 );

    // a part of the mesh: the triangles with x < 2
    FaceBitSet region( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
        if ( mesh.triCenter( f ).x < 2 )
            region.set( f );
    const auto part = fitBSplineSurfaceToMesh( { mesh, &region }, { .tolerance = 1e-3 } ).value();
    EXPECT_LT( part.maxError, 1e-3 );
    EXPECT_GT( part.distanceSq( surfPoint( 3, 1 ) ), 0.25 ); // 0.5 squared

    // not reachable tolerance: the surface with the most control points allowed for 651 vertices (40x8) or a better one
    const auto fine = fitBSplineSurfaceToMesh( mesh ).value();
    EXPECT_LE( fine.maxError, s.maxError );

    // triangles 5 times longer than the others at one side of the patch get additional points inside
    const auto xOf = []( size_t i ) { return i <= 40 ? 0.05 * i : 2 + 0.5 * ( i - 40 ); };
    const auto uneven = makeRegularGridMesh( 43, 21, []( size_t, size_t ) { return true; },
        [&]( size_t x, size_t y ) { return Vector3f( surfPoint( xOf( x ), 0.1 * y ) ); } ).value();
    const auto us = fitBSplineSurfaceToMesh( uneven, { .tolerance = 1e-3 } ).value();
    EXPECT_LT( us.maxError, 1e-3 );
    for ( auto f : uneven.topology.getValidFaces() )
    {
        const auto c = uneven.triCenter( f );
        if ( c.x > 2 ) // the triangles are far from the patch there, and the surface is closer to it
            EXPECT_LT( us.distanceSq( surfPoint( c.x, c.y ) ), 1e-4 ); // 0.01 squared
        EXPECT_GT( dot( us.surface.normal( us.project( Vector3d( c ) ) ), Vector3d( uneven.normal( f ) ) ), 0.9 );
    }

    // a square part takes all the domain inside the margin: its corners are near the corners (the conformal map of a curved square is not a square)
    const auto square = makeRegularGridMesh( 21, 21, []( size_t, size_t ) { return true; },
        [&]( size_t x, size_t y ) { return Vector3f( surfPoint( 0.1 * x, 0.1 * y ) ); } ).value();
    for ( double margin : { 0.06, 0.2 } )
    {
        const auto ss = fitBSplineSurfaceToMesh( square, { .tolerance = 1e-3, .margin = margin } ).value();
        EXPECT_LT( ss.maxError, 1e-3 );
        for ( const auto& corner : { surfPoint( 0, 0 ), surfPoint( 2, 0 ), surfPoint( 0, 2 ), surfPoint( 2, 2 ) } )
        {
            const auto uv = ss.project( Vector3d( Vector3f( corner ) ) );
            EXPECT_NEAR( std::min( uv.x, 1 - uv.x ), margin, 0.05 );
            EXPECT_NEAR( std::min( uv.y, 1 - uv.y ), margin, 0.05 );
        }
    }
    EXPECT_FALSE( fitBSplineSurfaceToMesh( square, { .margin = 0.5 } ).has_value() );
    EXPECT_FALSE( fitBSplineSurfaceToMesh( square, { .margin = -0.1 } ).has_value() );

    // too few vertices
    const auto small = makeRegularGridMesh( 5, 5, []( size_t, size_t ) { return true; },
        []( size_t x, size_t y ) { return Vector3f( float( x ), float( y ), 0 ); } ).value();
    EXPECT_FALSE( fitBSplineSurfaceToMesh( small ).has_value() );
}

} //namespace MR
