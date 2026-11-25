#include <MRMesh/MRBSpline.h>
#include <MRMesh/MRConformalMap.h>
#include <MRMesh/MRConstants.h>
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

TEST( MRMesh, LeastSquaresConformalMap )
{
    // a grid on an inclined plane, its conformal map is a similarity
    const int nx = 6, ny = 4;
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
    const auto uv = leastSquaresConformalMap( mesh );
    ASSERT_TRUE( uv.has_value() ) << uv.error();
    const VertId v0( 0 ), v1( nx * ny - 1 );
    const double scale = ( ( *uv )[v1] - ( *uv )[v0] ).length() / ( mesh.points[v1] - mesh.points[v0] ).length();
    for ( auto a : mesh.topology.getValidVerts() )
        for ( auto b : mesh.topology.getValidVerts() )
            EXPECT_NEAR( ( ( *uv )[a] - ( *uv )[b] ).length(), scale * ( mesh.points[a] - mesh.points[b] ).length(), 1e-5 );
}

} //namespace MR
