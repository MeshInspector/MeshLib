#include <MRMesh/MRAABBTree.h>
#include <MRMesh/MRAABBTreeMaker.h>
#include <MRMesh/MRAABBTreeWide.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshCollide.h>
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
    for ( NodeId n( 0 ); n < nodes.size(); ++n )
    {
        const auto & node = nodes[n];
        if ( node.leaf() )
        {
            const auto f = node.leafId();
            ASSERT_TRUE( mesh.topology.hasFace( f ) );
            EXPECT_FALSE( leaves.test_set( f ) );
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
}

// checks that every valid face of the mesh is in exactly one leaf with its vertices and a box containing it,
// and the box of every node contains the boxes of its triangles or of its children having larger ids
static void expectValidTree( const AABBTreeWide & tree, const Mesh & mesh )
{
    const auto & nodes = tree.nodes();
    EXPECT_EQ( tree.leafTris().size(), size_t( mesh.topology.numValidFaces() ) );
    ASSERT_EQ( tree.leafBoxes().size(), tree.leafTris().size() );
    FaceBitSet faces( mesh.topology.faceSize() );
    for ( NodeId n( 0 ); n < nodes.size(); ++n )
    {
        const auto & node = nodes[n];
        if ( node.leaf() )
        {
            ASSERT_TRUE( node.numLeafTris() > 0 && node.first + node.numLeafTris() <= tree.leafTris().size() );
            for ( int i = node.first; i < node.first + node.numLeafTris(); ++i )
            {
                const auto & lt = tree.leafTris()[i];
                ASSERT_TRUE( mesh.topology.hasFace( lt.face ) );
                EXPECT_FALSE( faces.test_set( lt.face ) );
                EXPECT_EQ( lt.verts, mesh.topology.getTriVerts( lt.face ) );
                for ( auto v : lt.verts )
                    EXPECT_TRUE( tree.leafBoxes()[i].contains( mesh.points[v] ) );
                EXPECT_TRUE( node.box.contains( tree.leafBoxes()[i] ) );
            }
        }
        else
        {
            for ( int i = 0; i < node.count; ++i )
            {
                const NodeId c( node.first + i );
                ASSERT_TRUE( c > n && c < nodes.size() );
                EXPECT_TRUE( node.box.contains( nodes[c].box ) );
            }
        }
    }
    EXPECT_EQ( faces.count(), size_t( mesh.topology.numValidFaces() ) );
}

// expects the same projections of some points on both meshes
static void expectSameProjections( const Mesh & a, const Mesh & b )
{
    const auto box = a.computeBoundingBox();
    for ( int i = 0; i < 1000; ++i )
    {
        const Vector3f pt = box.min + mult( box.size() * 1.4f, Vector3f( ( i * 37 % 101 ) / 100.0f, ( i * 53 % 103 ) / 102.0f, ( i * 71 % 107 ) / 106.0f ) ) - box.size() * 0.2f;
        EXPECT_EQ( findProjection( pt, a ).distSq, findProjection( pt, b ).distSq );
    }
}

TEST( MRMesh, AABBTreeWideRefit )
{
    Mesh mesh = makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    (void)mesh.getAABBTreeWide();

    // move some vertices
    VertBitSet changed( mesh.topology.vertSize() );
    for ( auto v : mesh.topology.getValidVerts() )
    {
        if ( mesh.points[v].x > 0.3f )
        {
            mesh.points[v] *= 1.2f;
            changed.set( v );
        }
    }
    mesh.updateCaches( &changed );
    const auto * tree = mesh.getAABBTreeWideNotCreate();
    ASSERT_TRUE( tree );
    expectValidTree( *tree, mesh );
    Mesh rebuilt = mesh;
    rebuilt.invalidateCaches();
    expectSameProjections( mesh, rebuilt );

    // move all vertices
    for ( auto & p : mesh.points )
        p.z *= 0.5f;
    mesh.updateCaches( nullptr );
    tree = mesh.getAABBTreeWideNotCreate();
    ASSERT_TRUE( tree );
    expectValidTree( *tree, mesh );
    rebuilt = mesh;
    rebuilt.invalidateCaches();
    expectSameProjections( mesh, rebuilt );
}

TEST( MRMesh, AABBTreeWidePackOptimally )
{
    Mesh mesh = makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    FaceBitSet del( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
        if ( mesh.triCenter( f ).y > 0.5f )
            del.set( f );
    mesh.deleteFaces( del );
    (void)mesh.getAABBTreeWide();
    mesh.packOptimally( true );
    ASSERT_EQ( mesh.topology.faceSize(), mesh.topology.numValidFaces() );
    const auto * tree = mesh.getAABBTreeWideNotCreate();
    ASSERT_TRUE( tree );
    expectValidTree( *tree, mesh );
    // the faces are numbered in the order of the leaves
    for ( size_t i = 0; i < tree->leafTris().size(); ++i )
        EXPECT_EQ( tree->leafTris()[i].face, FaceId( i ) );
    Mesh rebuilt = mesh;
    rebuilt.invalidateCaches();
    expectSameProjections( mesh, rebuilt );
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
    // the wide tree built by findCollidingTriangleBitsets above was updated too
    const auto * wideTree = mesh.getAABBTreeWideNotCreate();
    ASSERT_TRUE( wideTree );
    expectValidTree( *wideTree, mesh );

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

TEST(MRMesh, AABBTreeCopyDuringConstruction)
{
    Mesh mesh = makeUVSphere(); // use larger mesh to increase the probability of copying during construction
    tbb::task_group tasks;
    tasks.run( [&] { mesh.getAABBTree(); } ); // construct the tree
    tasks.run( [&] { Mesh( mesh ).getAABBTree(); } ); // copy the mesh, then construct the tree for it
    tasks.wait();
}

} //namespace MR
