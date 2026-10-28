#include <MRMesh/MRBestPlaneVolume.h>
#include <MRMesh/MRBestFit.h>
#include <MRMesh/MREnumNeighbours.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTorus.h>
#include <gtest/gtest.h>

namespace MR
{

namespace
{

VertScalars computeBestPlaneVolumesNaive( const Mesh& mesh, float radius )
{
    VertScalars res( mesh.topology.vertSize(), 0.0f );
    EnumNeihbourFaces e;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        std::vector<FaceId> region;
        PointAccumulator acc;
        e.run( mesh.topology, v, [&]( FaceId f )
        {
            if ( ( mesh.triCenter( f ) - mesh.points[v] ).lengthSq() > radius * radius )
                return false;
            region.push_back( f );
            acc.addPoint( mesh.triCenter( f ), mesh.area( f ) );
            return true;
        } );
        if ( !acc.valid() )
            continue;
        const auto bestPlane = acc.getBestPlane();
        double sum = 0;
        for ( auto f : region )
            sum += bestPlane.distance( Vector3d( mesh.triCenter( f ) ) ) * dot( bestPlane.n, 0.5 * Vector3d( mesh.dirDblArea( f ) ) );
        res[v] = float( sum / ( double( radius ) * radius * radius ) );
    }
    return res;
}

} //anonymous namespace

TEST( MRMesh, BestPlaneVolumesSphere )
{
    auto sphere = makeSphere( { .radius = 1.0f, .numMeshVertices = 3000 } );
    const float radius = 0.3f;
    const float expected = PI_F * radius * radius * radius / 48; // small radius approximation for unit sphere

    auto convex = computeBestPlaneVolumes( sphere, radius );
    ASSERT_TRUE( convex.has_value() );
    for ( auto v : sphere.topology.getValidVerts() )
        EXPECT_NEAR( ( *convex )[v], expected, 0.2f * expected );

    sphere.topology.flipOrientation();
    auto concave = computeBestPlaneVolumes( sphere, radius );
    ASSERT_TRUE( concave.has_value() );
    for ( auto v : sphere.topology.getValidVerts() )
        EXPECT_NEAR( ( *concave )[v], -( *convex )[v], 1e-6f );
}

TEST( MRMesh, BestPlaneVolumesNaive )
{
    const auto torus = makeTorus( 1.0f, 0.4f, 48, 24 );
    const float radius = 0.25f;
    const auto fast = computeBestPlaneVolumes( torus, radius );
    ASSERT_TRUE( fast.has_value() );
    const auto naive = computeBestPlaneVolumesNaive( torus, radius );
    for ( auto v : torus.topology.getValidVerts() )
        EXPECT_NEAR( ( *fast )[v], naive[v], 1e-6f );
}

} //namespace MR
