#include <MRMesh/MRSurfacePath.h>
#include <gtest/gtest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>

namespace MR
{

TEST( MRMesh, SurfacePathTargets )
{
    Triangulation t{
        { 0_v, 1_v, 2_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );

    mesh.points.emplace_back( 0.f, 0.f, 0.f ); // 0_v
    mesh.points.emplace_back( 1.f, 0.f, 0.f ); // 1_v
    mesh.points.emplace_back( 0.f, 1.f, 0.f ); // 2_v

    VertBitSet starts(3);
    starts.set( 1_v );
    starts.set( 2_v );

    VertBitSet ends(3);
    ends.set( 0_v );

    const auto map = computeClosestSurfacePathTargets( mesh, starts, ends );
    EXPECT_EQ( map.size(), starts.count() );
    for ( const auto & [start, end] : map )
    {
        EXPECT_TRUE( starts.test( start ) );
        EXPECT_TRUE( ends.test( end ) );
    }
}

TEST( MRMesh, GeodesicPathFastMarchingReachesEnd )
{
    // on this coarse torus, steepest descent in the fast-marching field gets stuck in a vertex far from the end
    const auto mesh = makeTorus( 1.0f, 0.3f, 8, 4 );
    const MeshTriPoint start( mesh.topology.edgeWithLeft( 1_f ), { 1 / 3.f, 1 / 3.f } );
    const MeshTriPoint end( mesh.topology.edgeWithLeft( 17_f ), { 0.375f, 0.5f } );

    const auto path = computeGeodesicPath( mesh, start, end, GeodesicPathApprox::FastMarching );
    ASSERT_TRUE( path.has_value() );
    ASSERT_FALSE( path->empty() );
    // every two consecutive points of start, path, end must share a triangle
    EXPECT_TRUE( fromSameTriangle( mesh.topology, MeshTriPoint( start ), MeshTriPoint( path->front() ) ) );
    for ( size_t i = 0; i + 1 < path->size(); ++i )
        EXPECT_TRUE( fromSameTriangle( mesh.topology, MeshTriPoint( ( *path )[i] ), MeshTriPoint( ( *path )[i + 1] ) ) );
    EXPECT_TRUE( fromSameTriangle( mesh.topology, MeshTriPoint( path->back() ), MeshTriPoint( end ) ) );
}

} //namespace MR
