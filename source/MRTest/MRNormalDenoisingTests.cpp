#include <MRMesh/MRNormalDenoising.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshNormals.h>
#include <gtest/gtest.h>

namespace MR
{

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

} //namespace MR
