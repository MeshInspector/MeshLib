#include <MRMesh/MRNormalDenoising.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MREdgeIterator.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshNormals.h>
#include <MRMesh/MRRegionBoundary.h>
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

TEST( MRMesh, MeshDenoiseViaNormalsInvalidatesCaches )
{
    Mesh sphere = makeUVSphere( 1, 16, 16 );
    sphere.getAABBTree();
    EXPECT_TRUE( sphere.getAABBTreeNotCreate() );

    EXPECT_TRUE( meshDenoiseViaNormals( sphere ).has_value() );
    EXPECT_FALSE( sphere.getAABBTreeNotCreate() );
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

TEST( MRMesh, MeshDenoiseWithCreasesRegion )
{
    const Mesh noisy = noisySphere();

    FaceBitSet region( noisy.topology.faceSize() );
    for ( auto f : noisy.topology.getValidFaces() )
        if ( noisy.triCenter( f ).z > 0 )
            region.set( f );
    const auto innerVerts = getRegionInnerVerts( noisy.topology, region );
    EXPECT_GT( innerVerts.count(), 0 );

    Mesh mesh = noisy;
    DenoiseWithCreasesSettings settings;
    settings.region = &region;
    meshDenoiseWithCreases( mesh, {}, settings );

    // only the inner vertices of the region move
    float maxShiftIn = 0, maxShiftOut = 0;
    for ( auto v : noisy.topology.getValidVerts() )
    {
        auto & maxShift = innerVerts.test( v ) ? maxShiftIn : maxShiftOut;
        maxShift = std::max( maxShift, ( mesh.points[v] - noisy.points[v] ).length() );
    }
    EXPECT_GT( maxShiftIn, 1e-3f );
    EXPECT_EQ( maxShiftOut, 0 );

    // the faces completely inside the region become smoother
    const auto roughness = [&]( const Mesh & m )
    {
        const auto normals = computePerFaceNormals( m );
        float res = 0;
        for ( auto ue : undirectedEdges( m.topology ) )
        {
            const EdgeId e = ue;
            const auto l = m.topology.left( e );
            const auto r = m.topology.right( e );
            if ( l && r && region.test( l ) && region.test( r ) )
                res += ( normals[l] - normals[r] ).lengthSq();
        }
        return res;
    };
    EXPECT_LT( roughness( mesh ), 0.5f * roughness( noisy ) );
}

TEST( MRMesh, MeshDenoiseWithCreasesProgress )
{
    const Mesh noisy = noisySphere();

    Mesh mesh = noisy;
    float last = -1;
    bool ordered = true;
    const auto res = meshDenoiseWithCreases( mesh, {}, {}, [&]( float p )
    {
        ordered = ordered && p >= last && p <= 1;
        last = p;
        return true;
    } );
    EXPECT_TRUE( res.has_value() );
    EXPECT_TRUE( ordered );
    EXPECT_EQ( last, 1.0f ); // the progress must reach the end

    // the same denoising without a callback must give the same points
    Mesh quiet = noisy;
    meshDenoiseWithCreases( quiet, {} );
    float maxDiff = 0;
    for ( auto v : noisy.topology.getValidVerts() )
        maxDiff = std::max( maxDiff, ( quiet.points[v] - mesh.points[v] ).length() );
    EXPECT_EQ( maxDiff, 0 );

    // canceling from the callback must leave an error
    Mesh canceled = noisy;
    EXPECT_FALSE( meshDenoiseWithCreases( canceled, {}, {}, []( float ) { return false; } ).has_value() );
}

} //namespace MR
