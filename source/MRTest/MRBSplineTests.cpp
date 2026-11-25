#include <MRMesh/MRBSpline.h>
#include <MRMesh/MRConformalMap.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshPart.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, InterpolateBSplineCurve )
{
    // a helix
    std::vector<Vector3d> pts;
    for ( int i = 0; i < 40; ++i )
        pts.emplace_back( std::cos( 0.2 * i ), std::sin( 0.2 * i ), 0.05 * i );
    const auto c = interpolateBSplineCurve( pts );
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

    const auto line = interpolateBSplineCurve( { Vector3d( 0, 0, 0 ), Vector3d( 2, 0, 0 ) } );
    EXPECT_EQ( line.degree, 1 );
    EXPECT_NEAR( ( line.eval( 0.25 ) - Vector3d( 0.5, 0, 0 ) ).length(), 0, 1e-15 );

    const std::vector<Vector3d> three = { Vector3d( 0, 0, 0 ), Vector3d( 1, 1, 0 ), Vector3d( 2, 0, 0 ) };
    const auto parabola = interpolateBSplineCurve( three );
    EXPECT_EQ( parabola.degree, 2 );
    for ( int i = 0; i < 3; ++i )
        EXPECT_NEAR( ( parabola.eval( i / 2.0 ) - three[i] ).length(), 0, 1e-15 );

    // uneven spacing: the parameters are proportional to the length of the polyline
    const std::vector<Vector3d> uneven = { Vector3d( 0, 0, 0 ), Vector3d( 0.01, 0, 0 ), Vector3d( 1, 1, 0 ), Vector3d( 1, 1, 2 ), Vector3d( 5, 1, 2 ) };
    const auto uc = interpolateBSplineCurve( uneven );
    std::vector<double> t( 1, 0.0 );
    for ( size_t i = 1; i < uneven.size(); ++i )
        t.push_back( t.back() + ( uneven[i] - uneven[i - 1] ).length() );
    for ( size_t i = 0; i < uneven.size(); ++i )
        EXPECT_NEAR( ( uc.eval( t[i] / t.back() ) - uneven[i] ).length(), 0, 1e-12 );
}

TEST( MRMesh, FitBSplineSurface )
{
    std::vector<Vector3d> pts;
    std::vector<Vector2d> uvs;
    for ( int i = 0; i <= 20; ++i )
        for ( int j = 0; j <= 20; ++j )
        {
            const double u = i / 20.0, v = j / 20.0;
            uvs.emplace_back( u, v );
            pts.emplace_back( 3 * u, 2 * v, 2 * u - v + 1 );
        }
    // B-splines reproduce linear functions exactly
    auto s = fitBSplineSurface( pts, uvs );
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
        std::vector<double> weights( wPts.size(), 1.0 );
        weights.back() = 0;
        const auto ws = fitBSplineSurface( wPts, wUvs, { .weights = &weights } );
        EXPECT_NEAR( ( ws.eval( wUvs.back() ) - Vector3d( 1.5, 1, 1.5 ) ).length(), 0, 1e-9 );
    }

    // the weights multiply the squared distances: the plane shifted by -3 with weight 1 and by +1 with weight 3 give the plane
    {
        auto wPts = pts;
        auto wUvs = uvs;
        std::vector<double> weights( 2 * pts.size(), 1.0 );
        for ( size_t i = 0; i < pts.size(); ++i )
        {
            wPts[i].z -= 3;
            wPts.push_back( pts[i] + Vector3d( 0, 0, 1 ) );
            wUvs.push_back( uvs[i] );
            weights[pts.size() + i] = 3;
        }
        const auto ws = fitBSplineSurface( wPts, wUvs, { .weights = &weights } );
        for ( size_t i = 0; i < pts.size(); ++i )
            EXPECT_NEAR( ( ws.eval( uvs[i] ) - pts[i] ).length(), 0, 1e-9 );
    }

    // the smoothing keeps bilinear functions of (u, v): such a surface with points only in an ellipse is reproduced in all the domain
    const auto bilinear = []( const Vector2d& uv ) { return Vector3d( 3 * uv.x, 2 * uv.y, 2 * uv.x - uv.y + uv.x * uv.y ); };
    std::vector<Vector3d> ePts;
    std::vector<Vector2d> eUvs;
    for ( const auto& uv : uvs )
    {
        if ( sqr( ( uv.x - 0.5 ) / 0.44 ) + sqr( ( uv.y - 0.5 ) / 0.3 ) > 1 )
            continue;
        eUvs.push_back( uv );
        ePts.push_back( bilinear( uv ) );
    }
    for ( const auto& [nu, nv] : { std::pair( 6, 4 ), std::pair( 10, 5 ) } )
    {
        const auto es = fitBSplineSurface( ePts, eUvs, { .numU = nu, .numV = nv } );
        EXPECT_EQ( es.numU(), nu );
        EXPECT_EQ( es.numV(), nv );
        for ( const auto& uv : { Vector2d( 0, 0 ), Vector2d( 1, 0 ), Vector2d( 0, 1 ), Vector2d( 1, 1 ), Vector2d( 0.3, 0.6 ) } )
            EXPECT_NEAR( ( es.eval( uv ) - bilinear( uv ) ).length(), 0, 1e-8 );
    }

    // a curved surface
    for ( size_t i = 0; i < pts.size(); ++i )
        pts[i].z = std::sin( 3 * uvs[i].x ) * std::cos( 2 * uvs[i].y );
    s = fitBSplineSurface( pts, uvs, { .numU = 8, .numV = 8 } );
    for ( size_t i = 0; i < pts.size(); ++i )
        EXPECT_NEAR( ( s.eval( uvs[i] ) - pts[i] ).length(), 0, 1e-3 );

    // projection of a point above the surface
    const Vector2d uv0( 0.3, 0.7 );
    const auto p = s.eval( uv0 ) + 0.01 * s.normal( uv0 );
    const auto uv = s.project( p, Vector2d( 0.5, 0.5 ) );
    EXPECT_NEAR( ( uv - uv0 ).length(), 0, 1e-9 );
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

    // a collapsed row of control points: the derivative along v is zero at u = 0
    s.controlPoints = { Vector3d( 0, 0, 0 ), Vector3d( 0, 0, 0 ), Vector3d( 1, 0, 0 ), Vector3d( 1, 1, 0 ) };
    EXPECT_NEAR( ( s.project( Vector3d( 0.5, 0.25, 0 ), Vector2d( 0, 0.5 ) ) - Vector2d( 0.5, 0.5 ) ).length(), 0, 1e-12 );
}

TEST( MRMesh, LeastSquaresConformalMap )
{
    // a grid on an inclined plane, the conformal map of its part with i <= 5 is a similarity
    const int nx = 8, ny = 4;
    VertCoords points;
    for ( int j = 0; j < ny; ++j )
        for ( int i = 0; i < nx; ++i )
            points.emplace_back( float( i ), 0.7f * j, 0.3f * i + 0.2f * j );
    Triangulation t;
    for ( int j = 0; j + 1 < ny; ++j )
        for ( int i = 0; i + 1 < nx; ++i )
        {
            const VertId v( j * nx + i );
            t.push_back( { v, v + 1, v + nx + 1 } );
            t.push_back( { v, v + nx + 1, v + nx } );
        }
    const auto mesh = Mesh::fromTriangles( std::move( points ), t );
    FaceBitSet region( t.size() );
    for ( int i = 0; i < int( t.size() ); ++i )
        region.set( FaceId( i ), i / 2 % ( nx - 1 ) < 5 );
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

    // vertex 3 is only in a degenerate triangle, so its coordinates are not defined
    VertCoords degPoints{ Vector3f( 0, 0, 0 ), Vector3f( 2, 0, 0 ), Vector3f( 1, 1, 0 ), Vector3f( 1, 0, 0 ) };
    Triangulation degTris{ { VertId( 0 ), VertId( 1 ), VertId( 2 ) }, { VertId( 1 ), VertId( 0 ), VertId( 3 ) } };
    EXPECT_FALSE( leastSquaresConformalMap( Mesh::fromTriangles( std::move( degPoints ), degTris ) ).has_value() );

    auto nanMesh = mesh;
    nanMesh.points[v0].z = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE( leastSquaresConformalMap( nanMesh ).has_value() );
}

} //namespace MR
