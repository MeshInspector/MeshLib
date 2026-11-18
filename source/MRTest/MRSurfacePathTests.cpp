#include <MRMesh/MRSurfacePath.h>
#include <gtest/gtest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>

namespace MR
{

TEST( MRMesh, FastMarchingPathReachesEnd )
{
    // on this coarse torus, a vertex next to the triangle with end can be closer to end than that triangle's vertices
    const auto mesh = makeTorus( 1.0f, 0.3f, 8, 4 );
    const float bary[5][2] = { { 1 / 3.f, 1 / 3.f }, { 0.375f, 0.5f }, { 0.1f, 0.2f }, { 0.6f, 0.25f }, { 0.5f, 0 } };
    const MeshTriPoint start( mesh.topology.edgeWithLeft( 1_f ), { 1 / 3.f, 1 / 3.f } );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        for ( const auto & b : bary )
        {
            const MeshTriPoint end( mesh.topology.edgeWithLeft( f ), { b[0], b[1] } );
            if ( fromSameTriangle( mesh.topology, MeshTriPoint( start ), MeshTriPoint( end ) ) )
                continue;
            const auto path = computeFastMarchingPath( mesh, start, end );
            ASSERT_TRUE( path.has_value() );
            ASSERT_FALSE( path->empty() );
            EXPECT_TRUE( fromSameTriangle( mesh.topology, MeshTriPoint( end ), MeshTriPoint( path->back() ) ) ) << "end face " << int( f );
        }
    }
}

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

} //namespace MR
