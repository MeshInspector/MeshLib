#include <MRMesh/MRSurfacePath.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMeshBuilder.h>

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

} //namespace MR
