#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRphmap.h>
#include <gtest/gtest.h>

namespace MR
{

namespace
{

// duplicates edge (eb): splits its left face and flips the new edge going to the vertex opposite to (eb)
void addMultipleEdge( Mesh & mesh, EdgeId eb )
{
    auto & t = mesh.topology;
    const auto c = t.dest( t.next( eb ) );
    const auto v = mesh.splitFace( t.left( eb ) );
    auto e = t.edgeWithOrg( v );
    while ( t.dest( e ) != c )
        e = t.next( e );
    t.flipEdge( e );
    mesh.invalidateCaches();
}

// a box with 4-unit edges and a cylindrical pocket cut in its top face;
// (rim) receives the edges around the pocket with a box face on the left and a pocket face on the right
Mesh makeBoxWithPocket( std::vector<EdgeId> & rim )
{
    auto box = makeCube( Vector3f( 100, 100, 52 ), Vector3f( -50, -50, -50 ) );
    subdivideMesh( box, { .maxEdgeLen = 4.0f, .maxEdgeSplits = 10'000'000 } );
    const auto tool = makeCylinder( 5.0f, 10.0f, 32 );
    const auto xf = AffineXf3f::translation( Vector3f( 20, 0, -4 ) );
    BooleanResultMapper mapper;
    auto res = boolean( box, tool, BooleanOperation::DifferenceAB, &xf, &mapper );
    EXPECT_TRUE( res.valid() );
    const auto toolFaces = mapper.map( tool.topology.getValidFaces(), BooleanResultMapper::MapObject::B );
    const auto & t = res.mesh.topology;
    for ( EdgeId e( 0 ); e < t.edgeSize(); ++e )
    {
        const auto l = t.left( e ), r = t.right( e );
        if ( l && r && !toolFaces.test( l ) && toolFaces.test( r ) )
            rim.push_back( e );
    }
    return std::move( res.mesh );
}

} // namespace

TEST( MRMesh, FixMultipleEdgesNew2Old )
{
    Mesh mesh = makeCube();
    subdivideMesh( mesh, { .maxEdgeLen = 0.3f, .maxEdgeSplits = 1000 } );
    for ( auto p : { Vector3f( 0.6f, 0.1f, 0.1f ), Vector3f( -0.6f, -0.1f, 0.1f ) } )
        addMultipleEdge( mesh, mesh.topology.edgeWithLeft( findProjection( p, mesh ).proj.face ) );
    const auto multipleEdges = findMultipleEdges( mesh.topology ).value();
    ASSERT_EQ( multipleEdges.size(), 2 );

    const auto faceSize0 = mesh.topology.faceSize();
    const auto vertSize0 = mesh.topology.vertSize();
    Mesh ref = mesh;
    fixMultipleEdges( ref, multipleEdges );
    FaceHashMap new2Old;
    fixMultipleEdges( mesh, multipleEdges, &new2Old );
    EXPECT_TRUE( mesh == ref );
    EXPECT_FALSE( hasMultipleEdges( mesh.topology ) );

    const auto & t = mesh.topology;
    for ( FaceId f( faceSize0 ); f < t.faceSize(); ++f )
        EXPECT_EQ( t.hasFace( f ), new2Old.contains( f ) );
    EXPECT_EQ( new2Old.size(), 4 );
    for ( const auto & [nf, of] : new2Old )
    {
        EXPECT_LT( of, faceSize0 );
        EXPECT_TRUE( t.hasFace( of ) );
        // a new face is a part of the old one, and they share an edge incident to the new vertex
        bool sharesNewEdge = false;
        for ( auto e : leftRing( t, nf ) )
            if ( t.right( e ) == of && ( t.org( e ) >= vertSize0 || t.dest( e ) >= vertSize0 ) )
                sharesNewEdge = true;
        EXPECT_TRUE( sharesNewEdge );
    }
}

TEST( MRMesh, FixMeshDegeneraciesPatchRegion )
{
    std::vector<EdgeId> rim;
    Mesh mesh = makeBoxWithPocket( rim );
    ASSERT_GE( rim.size(), 15 );
    // needle slivers on the rim: maxDeviation below is under the float precision of their coordinates,
    // so neither the decimation nor the subdivision fixes them, and the patch step replaces their neighborhood
    for ( size_t i = 5; i < 15; ++i )
        mesh.splitEdge( rim[i], mesh.edgePoint( rim[i], 1e-6f ) );
    mesh.invalidateCaches();
    EXPECT_GE( findDegenerateFaces( mesh, 1e4f ).value().count(), 10 );

    FaceBitSet region = mesh.topology.getValidFaces();
    EXPECT_TRUE( fixMeshDegeneracies( mesh, { .maxDeviation = 1e-8f, .tinyEdgeLength = 1e-8f, .region = &region,
        .mode = FixMeshDegeneraciesParams::Mode::RemeshPatch } ).has_value() );
    EXPECT_TRUE( findDegenerateFaces( mesh, 1e4f ).value().none() );
    // the patch subdivision splits some edges on the boundaries of the holes, and the other parts of the faces split there join the region too
    EXPECT_EQ( ( mesh.topology.getValidFaces() - region ).count(), 0 );
}

TEST( MRMesh, FixMeshDegeneraciesPatchWholeComponent )
{
    const auto cylinder = makeCylinder( 1.0f, 20.0f, 1024 );
    FixMeshDegeneraciesParams params{ .maxDeviation = 1e-6f, .tinyEdgeLength = 1e-7f, .criticalTriAspectRatio = 1e3f,
        .mode = FixMeshDegeneraciesParams::Mode::Remesh };
    // every side face is degenerate, the decimation and the subdivision fix only some of them,
    // and the neighborhood of the remaining degenerations covers the whole mesh
    EXPECT_EQ( findDegenerateFaces( cylinder, params.criticalTriAspectRatio ).value().count(), 2 * 1024 );
    Mesh remeshed = cylinder;
    EXPECT_TRUE( fixMeshDegeneracies( remeshed, params ).has_value() );
    EXPECT_TRUE( findDegenerateFaces( remeshed, params.criticalTriAspectRatio ).value().any() );

    // patching the whole mesh would delete it, leaving no hole to fill, so the patch step does not change it
    params.mode = FixMeshDegeneraciesParams::Mode::RemeshPatch;
    Mesh patched = cylinder;
    EXPECT_TRUE( fixMeshDegeneracies( patched, params ).has_value() );
    EXPECT_EQ( patched.topology.numValidFaces(), remeshed.topology.numValidFaces() );
    EXPECT_TRUE( patched == remeshed );
}

} //namespace MR
