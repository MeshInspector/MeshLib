#include <MRMesh/MRAABBTree.h>
#include <MRMesh/MRAABBTreeMaker.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRLine3.h>
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
    EXPECT_EQ( tree.rootNodeId(), NodeId( 0 ) );
    EXPECT_EQ( tree.box( tree.rootNodeId() ), sphere.computeBoundingBox().insignificantlyExpanded() );
    EXPECT_EQ( tree.boxes().size(), tree.numLeaves() + tree.nodes().size() );
    EXPECT_EQ( tree.getBoundingBox(), sphere.computeBoundingBox().insignificantlyExpanded() );
    EXPECT_FALSE( tree.isLeaf( tree[tree.rootNodeId()].l ) );
    EXPECT_FALSE( tree.isLeaf( tree[tree.rootNodeId()].r ) );
    EXPECT_EQ( tree.leaves().size(), sphere.topology.numValidFaces() );

    assert( !tree.empty() );
    auto m = std::move( tree );
    assert( tree.empty() );

    FaceBitSet fs;
    fs.autoResizeSet( 1_f );
    AABBTree smallerTree( { sphere, &fs } );
    EXPECT_EQ( smallerTree.nodes().size(), 0 );
    EXPECT_EQ( smallerTree.numLeaves(), 1 );
    ASSERT_TRUE( smallerTree.isLeaf( smallerTree.rootNodeId() ) );
    EXPECT_EQ( smallerTree.leafId( smallerTree.rootNodeId() ), 1_f );
    EXPECT_EQ( smallerTree.getBoundingBox(), sphere.computeBoundingBox( &fs ).insignificantlyExpanded() );
    EXPECT_EQ( smallerTree.leafVerts( smallerTree.rootNodeId() ), sphere.topology.getTriVerts( 1_f ) );
    EXPECT_EQ( smallerTree.getSubtreeLeaves( smallerTree.rootNodeId() ), fs );
    EXPECT_EQ( smallerTree.getSubtrees( 4 ), std::vector<NodeId>{ smallerTree.rootNodeId() } );
}

TEST(MRMesh, ProjectionToEmptyMesh)
{
    Vector3f p( 1.f, 2.f, 3.f );
    bool hasProjection = Mesh{}.projectPoint( p ).valid();
    EXPECT_FALSE( hasProjection );
}

// checks that every valid face of the mesh is in exactly one leaf with the vertices of the face,
// and the box of every node contains the boxes of its children having larger ids
static void expectValidTree( const AABBTree & tree, const Mesh & mesh )
{
    const auto & nodes = tree.nodes();
    EXPECT_EQ( tree.numLeaves(), size_t( mesh.topology.numValidFaces() ) );
    EXPECT_EQ( tree.boxes().size(), tree.numLeaves() + nodes.size() );
    FaceBitSet leaves( mesh.topology.faceSize() );
    auto checkLeaf = [&]( NodeId c )
    {
        const auto f = tree.leafId( c );
        ASSERT_TRUE( mesh.topology.hasFace( f ) );
        EXPECT_FALSE( leaves.test_set( f ) );
        EXPECT_EQ( tree.leafVerts( c ), mesh.topology.getTriVerts( f ) );
        Box3f box;
        for ( auto v : mesh.topology.getTriVerts( f ) )
            box.include( mesh.points[v] );
        EXPECT_EQ( tree.leafBox( c ), box.insignificantlyExpanded() );
    };
    if ( nodes.empty() && !tree.empty() )
        checkLeaf( tree.rootNodeId() );
    for ( NodeId n( 0 ); n < nodes.size(); ++n )
    {
        const auto & node = nodes[n];
        for ( auto c : { node.l, node.r } )
        {
            if ( tree.isLeaf( c ) )
                checkLeaf( c );
            else
                ASSERT_TRUE( c > n && c < nodes.size() );
            EXPECT_TRUE( tree.box( n ).contains( tree.box( c ) ) );
        }
    }
    EXPECT_EQ( leaves.count(), size_t( mesh.topology.numValidFaces() ) );
}

TEST( MRMesh, AABBTreeRefitAndPack )
{
    Mesh mesh = makeSphere( { .radius = 1.0f, .numMeshVertices = 1000 } );
    (void)mesh.getAABBTree();

    // move some vertices and refit the tree
    VertBitSet changed( mesh.topology.vertSize() );
    for ( VertId v( 0 ); v < mesh.topology.vertSize(); v += 7 )
    {
        mesh.points[v] *= 1.2f;
        changed.set( v );
    }
    mesh.updateCaches( &changed );
    expectValidTree( mesh.getAABBTree(), mesh );
    EXPECT_EQ( mesh.getAABBTree().getBoundingBox(), AABBTree( mesh ).getBoundingBox() );

    // delete some faces and pack the mesh preserving the tree
    FaceBitSet del( mesh.topology.faceSize() );
    for ( FaceId f( 0 ); f < mesh.topology.faceSize(); f += 5 )
        del.set( f );
    mesh.topology.deleteFaces( del );
    mesh.invalidateCaches();
    (void)mesh.getAABBTree();
    mesh.packOptimally( true );
    const auto & tree = mesh.getAABBTree();
    expectValidTree( tree, mesh );
    EXPECT_EQ( tree.numLeaves(), mesh.topology.faceSize() );

    // the faces are in the order of the leaves now
    AABBTree::LeafBMap map;
    map.b.resize( mesh.topology.faceSize() );
    tree.getLeafOrder( map );
    for ( FaceId f( 0 ); f < mesh.topology.faceSize(); ++f )
        EXPECT_EQ( map.b[f], f );
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
        expectValidTree( mesh.getAABBTree(), mesh );
        for ( const Vector3f pt : { Vector3f( 0.3f, 0.4f, 2.0f ), Vector3f( 3.3f, 0.4f, 2.0f ) } )
        {
            const auto proj = findProjection( pt, mesh );
            ASSERT_TRUE( proj.proj.face );
            EXPECT_LT( ( mesh.triPoint( proj.mtp ) - proj.proj.point ).length(), 1e-5f );
            const auto hit = rayMeshIntersect( mesh, Line3f( pt, Vector3f( 0.0f, 0.0f, -1.0f ) ) );
            ASSERT_TRUE( hit.proj.face );
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
