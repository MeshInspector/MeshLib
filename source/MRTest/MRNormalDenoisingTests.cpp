#include <MRMesh/MRNormalDenoising.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MREdgeIterator.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshNormals.h>
#include <gtest/gtest.h>

namespace MR
{

namespace
{

// the total squared difference of normals over all pairs of adjacent faces
float normalRoughness( const Mesh & mesh )
{
    const auto normals = computePerFaceNormals( mesh );
    float res = 0;
    for ( auto ue : undirectedEdges( mesh.topology ) )
    {
        const EdgeId e = ue;
        const auto l = mesh.topology.left( e );
        const auto r = mesh.topology.right( e );
        if ( l && r )
            res += ( normals[l] - normals[r] ).lengthSq();
    }
    return res;
}

Mesh noisySphere()
{
    Mesh res = makeUVSphere( 1, 16, 16 );
    // deterministic displacement of every vertex along its own direction
    for ( auto v : res.topology.getValidVerts() )
        res.points[v] *= 1 + 0.02f * ( ( ( int( v ) * 37 ) % 7 ) - 3 );
    return res;
}

} //anonymous namespace

TEST( MRMesh, DenoiseNormalsStrong )
{
    const Mesh sphere = makeUVSphere( 1, 16, 16 );
    const auto normals0 = computePerFaceNormals( sphere );
    const Vector<float, UndirectedEdgeId> v( sphere.topology.undirectedEdgeSize(), 1 );

    // strong smoothing of an already smooth normal field must not change it much
    auto normals = normals0;
    denoiseNormals( sphere, normals, v, 100 );
    for ( auto f : sphere.topology.getValidFaces() )
        EXPECT_GT( dot( normals[f], normals0[f] ), 0.9f );
}

TEST( MRMesh, MeshDenoiseWithCreasesAllSharp )
{
    const Mesh noisy = noisySphere();

    // every edge is a crease, so the normals have nothing to be smoothed with and the points stay put
    const UndirectedEdgeBitSet creases( noisy.topology.undirectedEdgeSize(), true );
    Mesh mesh = noisy;
    meshDenoiseWithCreases( mesh, creases );

    float maxShift = 0;
    for ( auto v : mesh.topology.getValidVerts() )
        maxShift = std::max( maxShift, ( mesh.points[v] - noisy.points[v] ).length() );
    EXPECT_LT( maxShift, 1e-5f );
}

TEST( MRMesh, MeshDenoiseWithCreasesNoneSharp )
{
    const Mesh noisy = noisySphere();

    Mesh mesh = noisy;
    meshDenoiseWithCreases( mesh, {} );

    EXPECT_LT( normalRoughness( mesh ), 0.5f * normalRoughness( noisy ) );
}

TEST( MRMesh, MeshDenoiseWithCreasesTopologyAndPoints )
{
    const Mesh noisy = noisySphere();
    const UndirectedEdgeBitSet creases = noisy.findCreaseEdges( PI_F / 6 );
    EXPECT_GT( creases.count(), 0 ); // the noise must produce some creases, or the test is trivial

    Mesh mesh = noisy;
    meshDenoiseWithCreases( mesh, creases );

    // the same denoising through the topology-and-points overload must give the same points
    VertCoords points = noisy.points;
    meshDenoiseWithCreases( noisy.topology, points, creases );

    float maxDiff = 0;
    for ( auto v : noisy.topology.getValidVerts() )
        maxDiff = std::max( maxDiff, ( points[v] - mesh.points[v] ).length() );
    EXPECT_EQ( maxDiff, 0 );
}

} //namespace MR
