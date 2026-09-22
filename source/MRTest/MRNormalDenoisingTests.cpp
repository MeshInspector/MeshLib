#include <MRMesh/MRNormalDenoising.h>
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

} //namespace MR
