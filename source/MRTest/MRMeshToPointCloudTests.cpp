#include <MRMesh/MRMeshToPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointsProject.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRMeshPart.h>
#include <MRMesh/MRTorus.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>

namespace MR
{

namespace
{

// the largest distance from a point of the mesh surface to the nearest point of the cloud;
// each triangle is probed in its center, in the middles of its sides and in random points
float maxSurfaceToCloudDist( const MeshPart & mp, const PointCloud & cloud, int randomProbesPerFace )
{
    std::mt19937 gen( 20260827 );
    std::uniform_real_distribution<float> uniform( 0, 1 );
    float res = 0;
    for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
    {
        Vector3f v[3];
        mp.mesh.getTriPoints( f, v );
        auto probe = [&]( float a, float b )
        {
            const auto p = v[0] + a * ( v[1] - v[0] ) + b * ( v[2] - v[0] );
            res = std::max( res, std::sqrt( findProjectionOnPoints( p, cloud ).distSq ) );
        };
        probe( 1 / 3.f, 1 / 3.f );
        probe( 0.5f, 0 );
        probe( 0, 0.5f );
        probe( 0.5f, 0.5f );
        for ( int i = 0; i < randomProbesPerFace; ++i )
        {
            auto a = uniform( gen );
            auto b = uniform( gen );
            if ( a + b > 1 )
            {
                a = 1 - a;
                b = 1 - b;
            }
            probe( a, b );
        }
    }
    return res;
}

// the largest distance from a point of the cloud to the sampled surface
float maxCloudToSurfaceDist( const MeshPart & mp, const PointCloud & cloud )
{
    float res = 0;
    for ( auto v : cloud.validPoints )
        res = std::max( res, std::sqrt( findProjection( cloud.points[v], mp ).distSq ) );
    return res;
}

} // anonymous namespace

TEST( MRMesh, MeshToDensePointCloud )
{
    // the triangles of a torus are far from equilateral, and the resolutions are not powers of two
    const auto mesh = makeTorus( 1.0f, 0.3f, 12, 10 );

    size_t lastNumPoints = 0;
    for ( const float radius : { 0.2f, 0.05f } )
    {
        const auto cloud = meshToDensePointCloud( mesh, radius );
        ASSERT_TRUE( cloud.has_value() );
        // all mesh vertices are in the cloud keeping their ids
        ASSERT_GE( cloud->points.size(), mesh.points.size() );
        EXPECT_EQ( cloud->validPoints.count(), cloud->points.size() );
        for ( auto v : mesh.topology.getValidVerts() )
            EXPECT_EQ( cloud->points[v], mesh.points[v] );
        // the smaller the radius, the denser the cloud
        EXPECT_GT( cloud->points.size(), lastNumPoints );
        lastNumPoints = cloud->points.size();

        // the main property: no ball of the radius can pass through the mesh without touching a point
        EXPECT_LE( maxSurfaceToCloudDist( mesh, *cloud, 16 ), radius );
        // and every point of the cloud is on the mesh
        EXPECT_LE( maxCloudToSurfaceDist( mesh, *cloud ), 1e-6f );

        // every point has a unit normal directed as the mesh normal in the same point
        ASSERT_EQ( cloud->normals.size(), cloud->points.size() );
        for ( auto v : cloud->validPoints )
        {
            EXPECT_NEAR( cloud->normals[v].length(), 1.f, 1e-5f );
            const auto proj = findProjection( cloud->points[v], mesh );
            EXPECT_GT( dot( cloud->normals[v], mesh.normal( proj.mtp ) ), 0.9f );
        }

        // the same cloud without normals
        const auto noNormals = meshToDensePointCloud( mesh, radius, false );
        ASSERT_TRUE( noNormals.has_value() );
        EXPECT_TRUE( noNormals->normals.empty() );
        EXPECT_EQ( noNormals->points.vec_, cloud->points.vec_ );
    }

    EXPECT_FALSE( meshToDensePointCloud( mesh, 0 ).has_value() );
}

// the deleted faces and the edges left without faces must not contribute any samples
TEST( MRMesh, MeshToDensePointCloudPartial )
{
    auto mesh = makeTorus( 1.0f, 0.3f, 12, 10 );
    FaceBitSet toDelete( mesh.topology.faceSize(), false );
    for ( auto f : mesh.topology.getValidFaces() )
        if ( int( f ) % 3 == 0 )
            toDelete.set( f );
    mesh.deleteFaces( toDelete );

    const float radius = 0.1f;
    const auto cloud = meshToDensePointCloud( mesh, radius );
    ASSERT_TRUE( cloud.has_value() );
    const auto full = meshToDensePointCloud( makeTorus( 1.0f, 0.3f, 12, 10 ), radius );
    ASSERT_TRUE( full.has_value() );
    EXPECT_LT( cloud->points.size(), full->points.size() ); // the deleted faces gave no samples
    EXPECT_LE( maxSurfaceToCloudDist( mesh, *cloud, 16 ), radius );
    EXPECT_LE( maxCloudToSurfaceDist( mesh, *cloud ), 1e-6f );
}

// only the faces of the given part are sampled, and only their surface has to be covered
TEST( MRMesh, MeshToDensePointCloudPart )
{
    const auto mesh = makeTorus( 1.0f, 0.3f, 12, 10 );
    FaceBitSet region( mesh.topology.faceSize(), false );
    for ( auto f : mesh.topology.getValidFaces() )
        if ( int( f ) % 3 != 0 )
            region.set( f );

    const float radius = 0.1f;
    const auto part = meshToDensePointCloud( { mesh, &region }, radius );
    const auto whole = meshToDensePointCloud( mesh, radius );
    ASSERT_TRUE( part.has_value() );
    ASSERT_TRUE( whole.has_value() );
    EXPECT_LT( part->points.size(), whole->points.size() );
    // the vertices of the faces outside the part are not valid points of the cloud
    EXPECT_LT( part->validPoints.count(), whole->validPoints.count() );
    // the part is covered, and no point of the cloud is away from it
    EXPECT_LE( maxSurfaceToCloudDist( { mesh, &region }, *part, 16 ), radius );
    EXPECT_LE( maxCloudToSurfaceDist( { mesh, &region }, *part ), 1e-6f );
}

// two triangles share a long edge, and the vertices opposite to it are close to it, so the balls
// around the four vertices cover both triangles and no sample is needed at all
TEST( MRMesh, MeshToDensePointCloudCloseApexes )
{
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } );
    t.push_back( { 1_v, 0_v, 3_v } );
    const auto mesh = Mesh::fromTriangles( VertCoords{
        Vector3f{ 0, 0, 0 }, Vector3f{ 10, 0, 0 }, Vector3f{ 5, 0.05f, 0 }, Vector3f{ 5, -0.05f, 0 } }, t );

    // the covering radius of both triangles is about 2.5, while the edge is 10 long
    const float radius = 3;
    const auto cloud = meshToDensePointCloud( mesh, radius );
    ASSERT_TRUE( cloud.has_value() );
    EXPECT_EQ( cloud->points.size(), 4 );
    EXPECT_LE( maxSurfaceToCloudDist( mesh, *cloud, 64 ), radius );

    // a smaller radius does require samples
    const auto dense = meshToDensePointCloud( mesh, 1.0f );
    ASSERT_TRUE( dense.has_value() );
    EXPECT_GT( dense->points.size(), 4 );
    EXPECT_LE( maxSurfaceToCloudDist( mesh, *dense, 64 ), 1.0f );
}

// a triangle with all vertices on one line has infinite circumradius, but it still requires
// a finite number of samples, because its own vertices cover it within 1/4 of its length
TEST( MRMesh, MeshToDensePointCloudDegenerate )
{
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } );
    const auto mesh = Mesh::fromTriangles( VertCoords{ Vector3f{ 0, 0, 0 }, Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 0, 0 } }, t );

    const auto cloud = meshToDensePointCloud( mesh, 0.1f );
    ASSERT_TRUE( cloud.has_value() );
    // the triangle is flat, so the samples of its longest side cover it: that side is divided
    // in 5 parts, and the 3 vertices with the 4 samples between them is all the cloud has
    EXPECT_EQ( cloud->points.size(), 7 );
    EXPECT_LE( maxSurfaceToCloudDist( mesh, *cloud, 64 ), 0.1f );
}

} //namespace MR
