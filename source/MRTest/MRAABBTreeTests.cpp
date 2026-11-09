#include <MRMesh/MRAABBTree.h>
#include <MRMesh/MRAABBTreeMaker.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRLine.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBuffer.h>
#include <gtest/gtest.h>
#include <MRPch/MRTBB.h>
#include <algorithm>

namespace MR
{

TEST(MRMesh, AABBTree)
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    AABBTree tree( sphere );
    EXPECT_EQ( tree.nodes().size(), getNumNodes( sphere.topology.numValidFaces() ) );
    EXPECT_EQ( tree[AABBTree::rootNodeId()].box, sphere.computeBoundingBox().insignificantlyExpanded() );
    EXPECT_TRUE( tree[AABBTree::rootNodeId()].l.valid() );
    EXPECT_TRUE( tree[AABBTree::rootNodeId()].r.valid() );

    assert( !tree.nodes().empty() );
    auto m = std::move( tree );
    assert( tree.nodes().empty() );

    FaceBitSet fs;
    fs.autoResizeSet( 1_f );
    AABBTree smallerTree( { sphere, &fs } );
    EXPECT_EQ( smallerTree.nodes().size(), 1 );
}

TEST(MRMesh, ProjectionToEmptyMesh)
{
    Vector3f p( 1.f, 2.f, 3.f );
    bool hasProjection = Mesh{}.projectPoint( p ).valid();
    EXPECT_FALSE( hasProjection );
}

// checks that every valid face of the mesh is in exactly one leaf, which box contains the face,
// and the box of every other node contains the boxes of its children having larger ids
static void expectValidTree( const AABBTree & tree, const Mesh & mesh )
{
    const auto & nodes = tree.nodes();
    EXPECT_EQ( tree.numLeaves(), size_t( mesh.topology.numValidFaces() ) );
    FaceBitSet leaves( mesh.topology.faceSize() );
    std::vector<bool> leafPoses( tree.leafTriangulation().size() );
    for ( NodeId n( 0 ); n < nodes.size(); ++n )
    {
        const auto & node = nodes[n];
        if ( node.leaf() )
        {
            const auto f = node.leafId();
            ASSERT_TRUE( mesh.topology.hasFace( f ) );
            EXPECT_FALSE( leaves.test_set( f ) );
            ASSERT_TRUE( node.leafPos() >= 0 && node.leafPos() < leafPoses.size() );
            EXPECT_FALSE( leafPoses[node.leafPos()] );
            leafPoses[node.leafPos()] = true;
            EXPECT_EQ( tree.leafTriVerts( node ), mesh.topology.getTriVerts( f ) );
            Vector3f a, b, c;
            mesh.getTriPoints( f, a, b, c );
            EXPECT_TRUE( node.box.contains( a ) && node.box.contains( b ) && node.box.contains( c ) );
        }
        else
        {
            ASSERT_TRUE( node.l > n && node.r > n && node.l < nodes.size() && node.r < nodes.size() );
            EXPECT_TRUE( node.box.contains( nodes[node.l].box ) );
            EXPECT_TRUE( node.box.contains( nodes[node.r].box ) );
        }
    }
    EXPECT_EQ( leaves.count(), size_t( mesh.topology.numValidFaces() ) );
    EXPECT_EQ( leafPoses.size(), leaves.count() );
}

TEST( MRMesh, AABBTreeAddSplitFaces )
{
    // the sphere of the Boolean benchmark and its rotated copy
    Mesh mesh = makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    Mesh other = mesh;
    other.transform( AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), 0.1f ) ) );

    (void)mesh.getAABBTree();
    const Mesh orgMesh = mesh; // shares the tree with mesh

    // nothing is split, so the tree is still shared
    mesh.updateCachesAfterSplits( {} );
    EXPECT_EQ( mesh.getAABBTreeNotCreate(), orgMesh.getAABBTreeNotCreate() );

    // split some faces colliding with other mesh, a face several times, and an edge
    const auto collidingFaces = findCollidingTriangleBitsets( mesh, other ).first;
    ASSERT_GT( collidingFaces.count(), 100 );
    FaceHashMap new2Old;
    int n = 0;
    for ( auto f : collidingFaces )
        if ( n++ % 10 == 0 )
            mesh.splitFace( f, nullptr, &new2Old );
    const auto lastFace = mesh.topology.lastValidFace();
    mesh.splitFace( lastFace, nullptr, &new2Old ); // a part of a split face
    mesh.splitEdge( mesh.topology.edgeWithLeft( lastFace ), nullptr, &new2Old );
    // the new vertex is far outside of the split face, so the boxes of its ancestors in the tree have to grow
    const auto farFace = collidingFaces.find_last();
    mesh.splitFace( farFace, mesh.triCenter( farFace ) + 0.5f * mesh.normal( farFace ), nullptr, &new2Old );

    mesh.updateCachesAfterSplits( new2Old );
    const auto * tree = mesh.getAABBTreeNotCreate();
    ASSERT_TRUE( tree );
    expectValidTree( *tree, mesh );
    expectValidTree( orgMesh.getAABBTree(), orgMesh );

    // the updated tree finds the same collisions as a new one
    Mesh rebuilt = mesh;
    rebuilt.invalidateCaches();
    auto updatedRes = findCollidingTriangles( mesh, other );
    auto rebuiltRes = findCollidingTriangles( rebuilt, other );
    std::sort( updatedRes.begin(), updatedRes.end() );
    std::sort( rebuiltRes.begin(), rebuiltRes.end() );
    EXPECT_EQ( updatedRes, rebuiltRes );
    const auto numWithNewFaces = std::count_if( updatedRes.begin(), updatedRes.end(),
        [sz = orgMesh.topology.faceSize()]( const FaceFace & ff ) { return ff.aFace >= sz; } );
    EXPECT_GT( numWithNewFaces, 0 );
}

TEST( MRMesh, AABBTreeTriangulation )
{
    Mesh mesh = makeUVSphere( 1, 16, 16 );
    FaceBitSet del( mesh.topology.faceSize() );
    for ( FaceId f = 0_f; f < del.size(); f += 7 )
        del.set( f );
    mesh.deleteFaces( del );
    expectValidTree( mesh.getAABBTree(), mesh );

    // the tree is kept, and its triangulation is renumbered
    mesh.packOptimally( true );
    ASSERT_TRUE( mesh.getAABBTreeNotCreate() );
    expectValidTree( *mesh.getAABBTreeNotCreate(), mesh );

}

TEST( MRMesh, AABBTreeFlippedOrientation )
{
    Mesh mesh = makeSphere( { .radius = 1.0f, .numMeshVertices = 200 } );
    const auto firstSphereEdges = mesh.topology.undirectedEdgeSize();
    Mesh other = mesh;
    other.transform( AffineXf3f::translation( Vector3f( 3.0f, 0.0f, 0.0f ) ) );
    mesh.addMesh( other );
    (void)mesh.getAABBTree();
    auto expectRightTriPoints = [&]
    {
        ASSERT_TRUE( mesh.getAABBTreeNotCreate() );
        expectValidTree( *mesh.getAABBTreeNotCreate(), mesh );
        for ( const Vector3f pt : { Vector3f( 0.3f, 0.4f, 2.0f ), Vector3f( 3.3f, 0.4f, 2.0f ) } )
        {
            const auto proj = findProjection( pt, mesh );
            ASSERT_TRUE( proj.proj.face );
            EXPECT_EQ( mesh.topology.left( proj.mtp.e ), proj.proj.face );
            EXPECT_LT( ( mesh.triPoint( proj.mtp ) - proj.proj.point ).length(), 1e-5f );
            const auto hit = rayMeshIntersect( mesh, Line3f( pt, Vector3f( 0.0f, 0.0f, -1.0f ) ) );
            ASSERT_TRUE( hit.proj.face );
            EXPECT_EQ( mesh.topology.left( hit.mtp.e ), hit.proj.face );
            EXPECT_LT( ( mesh.triPoint( hit.mtp ) - hit.proj.point ).length(), 1e-5f );
        }
    };
    mesh.flipOrientation(); // the tree is updated, not rebuilt
    expectRightTriPoints();
    UndirectedEdgeBitSet firstSphere( mesh.topology.undirectedEdgeSize() );
    for ( UndirectedEdgeId ue( 0 ); ue < firstSphereEdges; ++ue )
        firstSphere.set( ue );
    mesh.flipOrientation( &firstSphere );
    expectRightTriPoints();
}

TEST(MRMesh, AABBTreeCopyDuringConstruction)
{
    Mesh mesh = makeUVSphere(); // use larger mesh to increase the probability of copying during construction
    tbb::task_group tasks;
    tasks.run( [&] { mesh.getAABBTree(); } ); // construct the tree
    tasks.run( [&] { Mesh( mesh ).getAABBTree(); } ); // copy the mesh, then construct the tree for it
    tasks.wait();
}

} //namespace MR
