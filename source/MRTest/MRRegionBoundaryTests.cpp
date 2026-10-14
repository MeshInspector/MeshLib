#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshTopology.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRId.h>
#include <MRMesh/MRRingIterator.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, findLeftBoundary)
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    FaceBitSet faces;
    faces.autoResizeSet( 0_f );
    auto paths = findLeftBoundary( sphere.topology, faces );
    EXPECT_EQ( paths.size(), 1 );
    for ( const auto & path : paths )
    {
        for ( auto e : path )
        {
            EXPECT_EQ( sphere.topology.left( e ), 0_f );
            EXPECT_NE( sphere.topology.right( e ), 0_f );
        }
    }
}

TEST( MRMesh, findRightBoundary )
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    FaceBitSet faces;
    faces.autoResizeSet( 0_f );
    auto paths = findRightBoundary( sphere.topology, faces );
    EXPECT_EQ( paths.size(), 1 );
    for ( const auto& path : paths )
    {
        for ( auto e : path )
        {
            EXPECT_EQ( sphere.topology.right( e ), 0_f );
            EXPECT_NE( sphere.topology.left( e ), 0_f );
        }
    }
}

TEST( MRMesh, getRegionInnerVerts )
{
    Mesh sphere = makeUVSphere( 1, 16, 16 );

    // make a hole around the upper pole
    FaceBitSet hole( sphere.topology.faceSize() );
    for ( auto f : sphere.topology.getValidFaces() )
        if ( sphere.triCenter( f ).z > 0.8f )
            hole.set( f );
    EXPECT_GT( hole.count(), 0 );
    sphere.topology.deleteFaces( hole );

    FaceBitSet region( sphere.topology.faceSize() );
    for ( auto f : sphere.topology.getValidFaces() )
        if ( sphere.triCenter( f ).z > 0 )
            region.set( f );

    const auto verts = getRegionInnerVerts( sphere.topology, region );
    EXPECT_EQ( verts, getIncidentVerts( sphere.topology, region ) - getRegionBoundaryVerts( sphere.topology, region ) );

    // the vertices on the hole boundary are returned, unlike getInnerVerts
    const auto innerVerts = getInnerVerts( sphere.topology, region );
    EXPECT_TRUE( innerVerts.is_subset_of( verts ) );
    EXPECT_GT( verts.count(), innerVerts.count() );
}

// every function with a separate path for small sets must give what a test of every mesh element gives
TEST( MRMesh, RegionBoundarySmallAndLargeSets )
{
    Mesh sphere = makeUVSphere( 1, 32, 32 );
    const auto & t = sphere.topology;
    // a hole around the upper pole and a single deleted face
    FaceBitSet deleted( t.faceSize() );
    for ( auto f : t.getValidFaces() )
        if ( sphere.triCenter( f ).z > 0.9f )
            deleted.set( f );
    deleted.set( 700_f );
    sphere.topology.deleteFaces( deleted );

    auto check = [&]( const FaceBitSet & faces, const char * name )
    {
        SCOPED_TRACE( name );
        auto inFaces = [&]( FaceId f ) { return f && faces.test( f ); };

        VertBitSet incVerts( t.getValidVerts().size() ), innerVerts( t.getValidVerts().size() ),
            bdVerts( t.vertSize() ), regionInnerVerts( t.vertSize() );
        for ( auto v : t.getValidVerts() )
        {
            int in = 0, out = 0, missing = 0;
            for ( auto e : orgRing( t, v ) )
            {
                const auto l = t.left( e );
                if ( !l )
                    ++missing;
                else if ( inFaces( l ) )
                    ++in;
                else
                    ++out;
            }
            incVerts.set( v, in > 0 );
            innerVerts.set( v, out == 0 && missing == 0 );
            bdVerts.set( v, in > 0 && out > 0 );
            regionInnerVerts.set( v, in > 0 && out == 0 );
        }
        EXPECT_EQ( getIncidentVerts( t, faces ), incVerts );
        EXPECT_EQ( getInnerVerts( t, faces ), innerVerts );
        EXPECT_EQ( getRegionBoundaryVerts( t, faces ), bdVerts );
        EXPECT_EQ( getRegionInnerVerts( t, faces ), regionInnerVerts );

        UndirectedEdgeBitSet incEdges( t.undirectedEdgeSize() ), innerEdges( t.undirectedEdgeSize() ), bdInside( t.undirectedEdgeSize() );
        EdgeBitSet leftBd( t.edgeSize() ), leftBdInside( t.edgeSize() );
        for ( UndirectedEdgeId ue{ 0 }; ue < t.undirectedEdgeSize(); ++ue )
        {
            const auto l = t.left( ue ), r = t.right( ue );
            incEdges.set( ue, inFaces( l ) || inFaces( r ) );
            innerEdges.set( ue, inFaces( l ) && inFaces( r ) );
            bdInside.set( ue, l && r && inFaces( l ) != inFaces( r ) );
            for ( EdgeId e : { EdgeId( ue ), EdgeId( ue ).sym() } )
            {
                const bool bd = inFaces( t.left( e ) ) && !inFaces( t.right( e ) );
                leftBd.set( e, bd );
                leftBdInside.set( e, bd && t.right( e ) );
            }
        }
        EXPECT_EQ( getIncidentEdges( t, faces ), incEdges );
        EXPECT_EQ( getInnerEdges( t, faces ), innerEdges );
        EXPECT_EQ( findRegionBoundaryUndirectedEdgesInsideMesh( t, faces ), bdInside );
        EXPECT_EQ( findAllLeftBdEdges( t, &faces ), leftBd );
        EXPECT_EQ( findAllLeftBdEdges( t, &faces, true ), leftBdInside );

        // vertex and edge inputs: the vertices and edges of the region
        const auto verts = getIncidentVerts( t, faces );
        UndirectedEdgeBitSet innerEdgesOfVerts( t.undirectedEdgeSize() ), edgesNearEdges( t.undirectedEdgeSize() );
        for ( UndirectedEdgeId ue{ 0 }; ue < t.undirectedEdgeSize(); ++ue )
        {
            const auto o = t.org( ue ), d = t.dest( ue );
            innerEdgesOfVerts.set( ue, o && d && verts.test( o ) && verts.test( d ) );
            bool touches = incEdges.test( ue );
            for ( EdgeId e : { EdgeId( ue ), EdgeId( ue ).sym() } )
                for ( auto ei : orgRing( t, e ) )
                    touches = touches || incEdges.test( ei.undirected() );
            edgesNearEdges.set( ue, touches );
        }
        EXPECT_EQ( getInnerEdges( t, verts ), innerEdgesOfVerts );
        EXPECT_EQ( getIncidentEdges( t, incEdges ), edgesNearEdges );

        FaceBitSet incFaces( t.faceSize() ), innerFaces( t.faceSize() );
        for ( auto f : t.getValidFaces() )
        {
            int in = 0, all = 0;
            for ( auto e : leftRing( t, f ) )
            {
                ++all;
                in += verts.test( t.org( e ) );
            }
            incFaces.set( f, in > 0 );
            innerFaces.set( f, in == all );
        }
        EXPECT_EQ( getIncidentFaces( t, verts ), incFaces );
        EXPECT_EQ( getInnerFaces( t, verts ), innerFaces );
    };

    // small sets: a few faces next to the hole, the deleted face and an id past the last face
    FaceBitSet smallSet( t.faceSize() + 10 );
    for ( auto f : t.getValidFaces() )
        if ( sphere.triCenter( f ).z > 0.8f && smallSet.count() < 20 )
            smallSet.set( f );
    smallSet.set( 700_f );
    smallSet.set( FaceId( t.faceSize() + 5 ) );
    check( smallSet, "small" );

    // a large region takes the scans of the whole mesh
    FaceBitSet large( t.faceSize() );
    for ( auto f : t.getValidFaces() )
        if ( sphere.triCenter( f ).z > -0.2f )
            large.set( f );
    check( large, "large" );
}

} //namespace MR
