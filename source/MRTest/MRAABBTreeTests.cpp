#include <MRMesh/MRAABBTree.h>
#include <MRMesh/MRAABBTreeMaker.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRLine.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRBitCast.h>
#include <cmath>
#include <gtest/gtest.h>
#include <MRPch/MRTBB.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>

namespace MR
{

TEST(MRMesh, AABBTree)
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    AABBTree tree( sphere );
    EXPECT_EQ( tree.leafTris().size(), sphere.topology.numValidFaces() );
    EXPECT_EQ( tree.getBoundingBox(), sphere.computeBoundingBox().insignificantlyExpanded() );
    EXPECT_FALSE( tree.nodes().front().leaf() );

    assert( !tree.nodes().empty() );
    auto m = std::move( tree );
    assert( tree.nodes().empty() );

    FaceBitSet fs;
    fs.autoResizeSet( 1_f );
    AABBTree smallerTree( { sphere, &fs } );
    EXPECT_EQ( smallerTree.nodes().size(), 1 );
    EXPECT_EQ( smallerTree.leafTris().size(), 1 );
}

TEST(MRMesh, ProjectionToEmptyMesh)
{
    Vector3f p( 1.f, 2.f, 3.f );
    bool hasProjection = Mesh{}.projectPoint( p ).valid();
    EXPECT_FALSE( hasProjection );
}

// the cached tree of a mesh without triangles: refit and winding numbers (asserted in Debug before)
TEST( MRMesh, AABBTreeOfEmptyMesh )
{
    Mesh mesh;
    EXPECT_FALSE( mesh.getBoundingBox().valid() );
    mesh.updateCaches( nullptr );
    EXPECT_EQ( mesh.calcFastWindingNumber( Vector3f( 1.f, 2.f, 3.f ) ), 0.f );
    EXPECT_TRUE( mesh.getAABBTree().getNodesFromFaces( FaceBitSet{} ).none() );
}

// checks that every valid face of the mesh is in exactly one leaf with its vertices and a box containing it,
// and the box of every node contains the boxes of its triangles or of its children having larger ids
static void expectValidTree( const AABBTree & tree, const Mesh & mesh )
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
    (void)mesh.getAABBTree();

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
    const auto * tree = mesh.getAABBTreeNotCreate();
    ASSERT_TRUE( tree );
    expectValidTree( *tree, mesh );
    Mesh rebuilt = mesh;
    rebuilt.invalidateCaches();
    expectSameProjections( mesh, rebuilt );

    // move all vertices
    for ( auto & p : mesh.points )
        p.z *= 0.5f;
    mesh.updateCaches( nullptr );
    tree = mesh.getAABBTreeNotCreate();
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
    (void)mesh.getAABBTree();
    mesh.packOptimally( true );
    ASSERT_EQ( mesh.topology.faceSize(), mesh.topology.numValidFaces() );
    const auto * tree = mesh.getAABBTreeNotCreate();
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

// splitting a part of the last split face again and again puts its leaf one level deeper every time:
// the tree is rebuilt before it gets deeper than MaxDepth, for which the stacks of the traversals are sized
TEST( MRMesh, AABBTreeDeepSplits )
{
    auto treeDepth = []( const AABBTree & tree )
    {
        const auto & nodes = tree.nodes();
        std::vector<int> depth( nodes.size(), 0 );
        int res = 0;
        for ( NodeId n( 0 ); n < nodes.size(); ++n )
        {
            if ( nodes[n].leaf() )
                continue;
            for ( int i = 0; i < nodes[n].count; ++i )
                res = std::max( res, depth[nodes[n].first + i] = depth[n] + 1 );
        }
        return res;
    };
    Mesh mesh = makeSphere( { .radius = 1.0f, .numMeshVertices = 1000 } );
    (void)mesh.getAABBTree();
    for ( int k = 0; k < 3 * AABBTree::MaxDepth; ++k )
    {
        FaceHashMap new2Old;
        mesh.splitFace( mesh.topology.lastValidFace(), nullptr, &new2Old );
        mesh.updateCachesAfterSplits( new2Old );
        const auto * tree = mesh.getAABBTreeNotCreate();
        ASSERT_TRUE( tree );
        EXPECT_LE( treeDepth( *tree ), AABBTree::MaxDepth );
    }
    expectValidTree( mesh.getAABBTree(), mesh );
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

// the leaf box of every triangle is the same as by Box::include and insignificantlyExpanded, also for zeros, denormals, extreme and non-finite coordinates
TEST( MRMesh, AABBTreeLeafBoxesOfSpecialValues )
{
    // no signaling NaN: Box::include of it depends on the compiler (clang for Arm turns a comparison with a constant into fminnm, which quiets it)
    const float values[] = { 0.f, -0.f, std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(), 1.f, -1.f,
        std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN() };
    constexpr int numValues = int( std::size( values ) );
    VertCoords points;
    Triangulation t;
    for ( int axis = 0; axis < 3; ++axis )
        for ( int i = 0; i < numValues; ++i )
            for ( int j = 0; j < numValues; ++j )
                for ( int k = 0; k < numValues; ++k )
                {
                    const float vs[3] = { values[i], values[j], values[k] };
                    if ( std::isnan( vs[0] ) && std::isnan( vs[1] ) && std::isnan( vs[2] ) )
                        continue; // an invalid box, which insignificantlyExpanded asserts against
                    const VertId v0( points.size() );
                    for ( int n = 0; n < 3; ++n )
                    {
                        // ordinary other coordinates: only the tested one decides between the SIMD and the scalar code
                        Vector3f p{ float( n + 1 ), float( i + 1 ), float( j + 1 ) };
                        p[axis] = vs[n];
                        points.push_back( p );
                    }
                    t.push_back( { v0, v0 + 1, v0 + 2 } );
                }
    const Mesh mesh = Mesh::fromTriangles( std::move( points ), t );
    const AABBTree tree( mesh );
    ASSERT_EQ( tree.leafTris().size(), t.size() );
    int numDiffer = 0;
    for ( size_t n = 0; n < tree.leafTris().size(); ++n )
    {
        const auto & lt = tree.leafTris()[n];
        // what Box::include gives, with NaN skipped explicitly: MSVC for arm64 compiles Box::include here with NaN kept
        Box3f box;
        for ( auto v : lt.verts )
            for ( int k = 0; k < 3; ++k )
            {
                const float x = mesh.points[v][k];
                if ( std::isnan( x ) )
                    continue;
                if ( x < box.min[k] )
                    box.min[k] = x;
                if ( box.max[k] < x )
                    box.max[k] = x;
            }
        const Box3f expected = box.insignificantlyExpanded();
        const Box3f & actual = tree.leafBoxes()[n];
        // compares the bits, NaN included
        if ( std::memcmp( &expected, &actual, sizeof( Box3f ) ) == 0 || ++numDiffer > 5 )
            continue;
        std::ostringstream s;
        s << std::hexfloat << "face " << int( lt.face ) << ", vertices";
        for ( auto v : lt.verts )
            s << " (" << mesh.points[v].x << ' ' << mesh.points[v].y << ' ' << mesh.points[v].z << ')';
        for ( const Box3f * b : { &expected, &actual } )
            s << ( b == &expected ? "\n  expected" : "\n  actual  " ) << " (" << b->min.x << ' ' << b->min.y << ' ' << b->min.z << ")-(" << b->max.x << ' ' << b->max.y << ' ' << b->max.z << ')';
        ADD_FAILURE() << s.str();
    }
    EXPECT_EQ( numDiffer, 0 );
}

namespace
{

/// a triangle soup with pseudo-random integer coordinates in [0, range): exactly representable, so the tree is the same on all platforms;
/// farOutlier moves the first vertex far away, which puts almost all centers in one bucket
Mesh makeIntegerSoup( int numTris, int range, bool farOutlier )
{
    VertCoords points;
    Triangulation t;
    std::uint32_t s = 1;
    const auto next = [&]
    {
        s = s * 1664525u + 1013904223u;
        return float( ( s >> 8 ) % std::uint32_t( range ) );
    };
    for ( int f = 0; f < numTris; ++f )
    {
        const VertId v0( points.size() );
        for ( int n = 0; n < 3; ++n )
        {
            const float x = next(), y = next(), z = next();
            points.push_back( { x, y, z } );
        }
        t.push_back( { v0, v0 + 1, v0 + 2 } );
    }
    if ( farOutlier )
        points.front() = Vector3f::diagonal( 16777216.f );
    return Mesh::fromTriangles( std::move( points ), t );
}

/// FNV-1a of the nodes, the leaf triangles and the leaf boxes of the tree
std::uint64_t treeHash( const AABBTree & tree )
{
    std::uint64_t h = 14695981039346656037ull;
    const auto add = [&h]( std::uint32_t x )
    {
        for ( int k = 0; k < 4; ++k )
        {
            h ^= ( x >> ( 8 * k ) ) & 0xff;
            h *= 1099511628211ull;
        }
    };
    const auto addBox = [&add]( const Box3f & b )
    {
        for ( int k = 0; k < 3; ++k )
        {
            add( bit_cast<std::uint32_t>( b.min[k] ) );
            add( bit_cast<std::uint32_t>( b.max[k] ) );
        }
    };
    for ( const auto & n : tree.nodes() )
    {
        addBox( n.box );
        add( std::uint32_t( n.first ) );
        add( std::uint32_t( n.count ) );
    }
    for ( const auto & lt : tree.leafTris() )
    {
        add( std::uint32_t( int( lt.face ) ) );
        for ( auto v : lt.verts )
            add( std::uint32_t( int( v ) ) );
    }
    for ( const auto & b : tree.leafBoxes() )
        addBox( b );
    return h;
}

} //anonymous namespace

// the trees of meshes with exactly representable coordinates are pinned by their hashes, the same as the tree maker before ML#7479 gave:
// small ranges, many equal centers, almost all centers in one bucket (split again in parallel for 150000 triangles), the parallel splits
TEST( MRMesh, AABBTreeGoldenHashes )
{
    struct Case { int numTris, range; bool farOutlier; std::uint64_t hash; };
    const Case cases[] = {
        { 1000, 1 << 20, false, 0x34b0ae3e1963543dull },
        { 20000, 4, false, 0x35cd81ed6d39cd05ull },
        { 150000, 1024, true, 0xb4c853ba04bdc510ull },
        { 300000, 1 << 16, false, 0x0cc9a28f326dd6dcull }
    };
    for ( const auto & c : cases )
    {
        const Mesh mesh = makeIntegerSoup( c.numTris, c.range, c.farOutlier );
        EXPECT_EQ( treeHash( AABBTree( mesh ) ), c.hash ) << c.numTris << " triangles";
        EXPECT_EQ( treeHash( AABBTree( mesh.points, mesh.topology.getTriangulation(), &mesh.topology.getValidFaces() ) ), c.hash )
            << c.numTris << " triangles from the triangulation";
    }
}

// the tree of a part of the triangles from the triangulation is the tree of that mesh part
TEST( MRMesh, AABBTreeFromTriangulationRegion )
{
    const Mesh mesh = makeUVSphere( 1.f, 64, 32 );
    FaceBitSet region( mesh.topology.faceSize() );
    for ( FaceId f : mesh.topology.getValidFaces() )
        region.set( f, f % 3 != 0 );
    EXPECT_EQ( treeHash( AABBTree( mesh.points, mesh.topology.getTriangulation(), &region ) ), treeHash( AABBTree( { mesh, &region } ) ) );
    EXPECT_TRUE( AABBTree( mesh.points, mesh.topology.getTriangulation(), &region ).leafTris().size() == region.count() );
    EXPECT_TRUE( AABBTree( mesh.points, Triangulation{} ).nodes().empty() );
}

// a big mesh, whose top ranges are split in parallel, gives the same tree on one thread
TEST( MRMesh, AABBTreeSameOnOneThread )
{
    const Mesh mesh = makeUVSphere( 1.f, 400, 200 ); // 160000 triangles: at least 2^17 for the parallel splits
    const AABBTree all( mesh );
    std::optional<AABBTree> one;
    tbb::task_arena( 1 ).execute( [&] { one.emplace( mesh ); } );
    ASSERT_EQ( all.nodes().size(), one->nodes().size() );
    for ( NodeId n( 0 ); n < all.nodes().size(); ++n )
    {
        const auto & a = all.nodes()[n];
        const auto & b = one->nodes()[n];
        ASSERT_TRUE( a.box == b.box && a.first == b.first && a.count == b.count ) << "node " << int( n );
    }
    ASSERT_EQ( all.leafTris().size(), one->leafTris().size() );
    for ( size_t i = 0; i < all.leafTris().size(); ++i )
    {
        const auto & a = all.leafTris()[i];
        const auto & b = one->leafTris()[i];
        ASSERT_TRUE( a.face == b.face && a.verts == b.verts && all.leafBoxes()[i] == one->leafBoxes()[i] ) << "leaf " << i;
    }
}

} //namespace MR
