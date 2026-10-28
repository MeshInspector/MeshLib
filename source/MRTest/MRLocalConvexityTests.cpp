#include <MRMesh/MRLocalConvexity.h>
#include <MRMesh/MRBestFit.h>
#include <MRMesh/MREnumNeighbours.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshToPointCloud.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointCloudRadius.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRTorus.h>
#include <gtest/gtest.h>

namespace MR
{

namespace
{

VertScalars computeLocalConvexityNaive( const Mesh& mesh, float radius )
{
    VertScalars res( mesh.topology.vertSize(), 0.0f );
    EnumNeihbourFaces e;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        std::vector<FaceId> region;
        FaceBitSet inRegion( mesh.topology.faceSize() );
        e.run( mesh.topology, v, [&]( FaceId f )
        {
            if ( ( mesh.triCenter( f ) - mesh.points[v] ).lengthSq() > radius * radius )
                return false;
            region.push_back( f );
            inRegion.set( f );
            return true;
        } );
        VertBitSet bdVerts( mesh.topology.vertSize() );
        for ( auto f : region )
        {
            for ( auto edge : leftRing( mesh.topology, f ) )
            {
                const auto r = mesh.topology.right( edge );
                if ( r && inRegion.test( r ) )
                    continue;
                bdVerts.set( mesh.topology.org( edge ) );
                bdVerts.set( mesh.topology.dest( edge ) );
            }
        }
        PointAccumulator acc;
        for ( auto bv : bdVerts )
            acc.addPoint( mesh.points[bv] );
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

TEST( MRMesh, LocalConvexitySphere )
{
    auto sphere = makeSphere( { .radius = 1.0f, .numMeshVertices = 3000 } );
    const float radius = 0.3f;
    const float expected = PI_F * radius / 4; // small radius approximation for unit sphere

    auto convex = computeLocalConvexity( sphere, radius );
    ASSERT_TRUE( convex.has_value() );
    for ( auto v : sphere.topology.getValidVerts() )
        EXPECT_NEAR( ( *convex )[v], expected, 0.2f * expected );

    sphere.topology.flipOrientation();
    auto concave = computeLocalConvexity( sphere, radius );
    ASSERT_TRUE( concave.has_value() );
    for ( auto v : sphere.topology.getValidVerts() )
        EXPECT_NEAR( ( *concave )[v], -( *convex )[v], 1e-6f );
}

TEST( MRMesh, LocalConvexityNaive )
{
    const auto torus = makeTorus( 1.0f, 0.4f, 48, 24 );
    const float radius = 0.25f;
    const auto fast = computeLocalConvexity( torus, radius );
    ASSERT_TRUE( fast.has_value() );
    const auto naive = computeLocalConvexityNaive( torus, radius );
    for ( auto v : torus.topology.getValidVerts() )
        EXPECT_NEAR( ( *fast )[v], naive[v], 1e-6f );
}

TEST( MRMesh, LocalConvexityPointCloudSphere )
{
    auto cloud = meshToPointCloud( makeSphere( { .radius = 1.0f, .numMeshVertices = 20000 } ) );
    const float radius = 0.3f;
    const float expected = PI_F * radius / 4; // same as for the mesh

    auto convex = computeLocalConvexity( cloud, radius );
    ASSERT_TRUE( convex.has_value() );
    for ( auto v : cloud.validPoints )
        EXPECT_NEAR( ( *convex )[v], expected, 0.2f * expected );

    for ( auto& n : cloud.normals )
        n = -n;
    auto concave = computeLocalConvexity( cloud, radius );
    ASSERT_TRUE( concave.has_value() );
    for ( auto v : cloud.validPoints )
        EXPECT_NEAR( ( *concave )[v], -( *convex )[v], 1e-6f );

    cloud.normals.clear();
    EXPECT_FALSE( computeLocalConvexity( cloud, radius ).has_value() );
}

TEST( MRMesh, LocalConvexityPointCloudNaive )
{
    const auto cloud = meshToPointCloud( makeTorus( 1.0f, 0.4f, 96, 48 ) );
    const float radius = 0.25f;
    const float shell = findAvgPointsRadius( cloud, 3 );
    const auto fast = computeLocalConvexity( cloud, radius );
    ASSERT_TRUE( fast.has_value() );
    for ( auto v : cloud.validPoints )
    {
        std::vector<VertId> ball;
        PointAccumulator acc;
        for ( auto u : cloud.validPoints )
        {
            const auto dist = ( cloud.points[u] - cloud.points[v] ).length();
            if ( dist <= radius )
                ball.push_back( u );
            if ( dist >= radius - shell / 2 && dist <= radius + shell / 2 )
                acc.addPoint( cloud.points[u] );
        }
        const auto plane = acc.getBestPlane();
        double sum = 0, sumAbsCos = 0;
        for ( auto u : ball )
        {
            sum += plane.distance( Vector3d( cloud.points[u] ) ) * dot( plane.n, Vector3d( cloud.normals[u] ) );
            sumAbsCos += std::abs( dot( plane.n, Vector3d( cloud.normals[u] ) ) );
        }
        EXPECT_NEAR( ( *fast )[v], PI * sum / ( sumAbsCos * radius ), 1e-5 );
    }
}

} //namespace MR
