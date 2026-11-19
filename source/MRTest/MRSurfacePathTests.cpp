#include <MRMesh/MRSurfacePath.h>
#include <gtest/gtest.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshPart.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRRingIterator.h>
#include <algorithm>

namespace MR
{

TEST( MRMesh, FastMarchingPathReachesEnd )
{
    // on coarse meshes, linear interpolation of exact distances near end can have local minima outside the triangle with end
    const float bary[5][2] = { { 1 / 3.f, 1 / 3.f }, { 0.375f, 0.5f }, { 0.1f, 0.2f }, { 0.6f, 0.25f }, { 0.5f, 0 } };
    for ( const auto & mesh : { makeTorus( 1.0f, 0.3f, 8, 4 ), makeTorus( 1.0f, 0.3f, 4, 24 ) } )
    {
        const MeshTriPoint start( mesh.topology.edgeWithLeft( 1_f ), { 1 / 3.f, 1 / 3.f } );
        for ( auto f : mesh.topology.getValidFaces() )
        {
            for ( const auto & b : bary )
            {
                const MeshTriPoint end( mesh.topology.edgeWithLeft( f ), { b[0], b[1] } );
                if ( fromSameTriangle( mesh.topology, MeshTriPoint( start ), MeshTriPoint( end ) ) )
                    continue;
                const auto path = computeFastMarchingPath( mesh, start, end );
                ASSERT_TRUE( path.has_value() );
                ASSERT_FALSE( path->empty() );
                EXPECT_TRUE( fromSameTriangle( mesh.topology, MeshTriPoint( end ), MeshTriPoint( path->back() ) ) ) << "end face " << int( f );
            }
        }
    }
}

TEST( MRMesh, FastMarchingPathInRegion )
{
    // the paths are limited to the half of a torus with thin triangles, given by faces or by vertices
    const auto mesh = makeTorus( 1.0f, 0.3f, 6, 24 );
    FaceBitSet faces( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
        if ( mesh.triCenter( f ).x > 0 )
            faces.set( f );
    const auto verts = getIncidentVerts( mesh.topology, faces );
    // the triangles with all vertices in verts: the paths can pass them even if the region is given by faces
    FaceBitSet vertFaces( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        const auto vs = mesh.topology.getTriVerts( f );
        if ( verts.test( vs[0] ) && verts.test( vs[1] ) && verts.test( vs[2] ) )
            vertFaces.set( f );
    }
    // the triangles containing given point
    auto trisWith = [&]( const MeshTriPoint & p )
    {
        std::vector<FaceId> res;
        if ( auto v = p.inVertex( mesh.topology ) )
        {
            for ( auto e : orgRing( mesh.topology, v ) )
                if ( auto l = mesh.topology.left( e ) )
                    res.push_back( l );
        }
        else if ( auto ep = p.onEdge( mesh.topology ) )
        {
            for ( auto l : { mesh.topology.left( ep.e ), mesh.topology.right( ep.e ) } )
                if ( l )
                    res.push_back( l );
        }
        else
            res.push_back( mesh.topology.left( p.e ) );
        return res;
    };
    // every two consecutive points among start, path and end share a triangle from the region
    auto insideRegion = [&]( const MeshTriPoint & start, const SurfacePath & path, const MeshTriPoint & end, const FaceBitSet & region ) -> bool
    {
        std::vector<MeshTriPoint> pts{ start };
        for ( const auto & ep : path )
            pts.emplace_back( ep );
        pts.push_back( end );
        for ( size_t i = 0; i + 1 < pts.size(); ++i )
        {
            const auto t0 = trisWith( pts[i] ), t1 = trisWith( pts[i + 1] );
            if ( !std::any_of( t0.begin(), t0.end(), [&]( FaceId f ) { return region.test( f ) && std::find( t1.begin(), t1.end(), f ) != t1.end(); } ) )
                return false;
        }
        return true;
    };

    const auto start = MeshTriPoint( mesh.topology.edgeWithLeft( faces.find_first() ), { 1 / 3.f, 1 / 3.f } );
    for ( auto f : faces )
    {
        const MeshTriPoint end( mesh.topology.edgeWithLeft( f ), { 0.2f, 0.3f } );
        if ( fromSameTriangle( mesh.topology, MeshTriPoint( start ), MeshTriPoint( end ) ) )
            continue;
        const auto byFaces = computeFastMarchingPath( { mesh, &faces }, start, end );
        const auto byVerts = computeFastMarchingPath( mesh, start, end, &verts );
        for ( const auto * path : { &byFaces, &byVerts } )
        {
            ASSERT_TRUE( path->has_value() );
            ASSERT_FALSE( ( *path )->empty() );
            EXPECT_TRUE( fromSameTriangle( mesh.topology, MeshTriPoint( end ), MeshTriPoint( ( *path )->back() ) ) ) << "end face " << int( f );
            EXPECT_TRUE( insideRegion( start, **path, end, vertFaces ) ) << "end face " << int( f );
        }
    }
}

TEST( MRMesh, SurfacePathTargets )
{
    Triangulation t{
        { 0_v, 1_v, 2_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );

    mesh.points.emplace_back( 0.f, 0.f, 0.f ); // 0_v
    mesh.points.emplace_back( 1.f, 0.f, 0.f ); // 1_v
    mesh.points.emplace_back( 0.f, 1.f, 0.f ); // 2_v

    VertBitSet starts(3);
    starts.set( 1_v );
    starts.set( 2_v );

    VertBitSet ends(3);
    ends.set( 0_v );

    const auto map = computeClosestSurfacePathTargets( mesh, starts, ends );
    EXPECT_EQ( map.size(), starts.count() );
    for ( const auto & [start, end] : map )
    {
        EXPECT_TRUE( starts.test( start ) );
        EXPECT_TRUE( ends.test( end ) );
    }
}

TEST( MRMesh, FastMarchingPathFlatTriangles )
{
    // on this symmetric torus some triangles get equal distances in all vertices,
    // and the steepest descent inside of such triangle has no direction
    const auto mesh = makeTorus( 1.0f, 0.3f, 3, 3 );
    // the regions without one vertex, given by vertices and by faces
    VertBitSet verts = mesh.topology.getValidVerts();
    verts.reset( 0_v );
    FaceBitSet faces( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        const auto vs = mesh.topology.getTriVerts( f );
        if ( verts.test( vs[0] ) && verts.test( vs[1] ) && verts.test( vs[2] ) )
            faces.set( f );
    }
    const VertBitSet * vertRegions[] = { nullptr, &verts, nullptr };
    const FaceBitSet * faceRegions[] = { nullptr, nullptr, &faces };
    for ( int i = 0; i < 3; ++i )
    {
        const MeshPart mp( mesh, faceRegions[i] );
        for ( auto f : faces )
        {
            const MeshTriPoint start( mesh.topology.edgeWithLeft( f ), { 1 / 3.f, 1 / 3.f } );
            for ( auto v : verts )
            {
                const MeshTriPoint end( mesh.topology, v );
                if ( fromSameTriangle( mesh.topology, MeshTriPoint( start ), MeshTriPoint( end ) ) )
                    continue;
                const auto fm = computeFastMarchingPath( mp, start, end, vertRegions[i] );
                ASSERT_TRUE( fm.has_value() ) << "region " << i << " face " << int( f ) << " vert " << int( v );
                EXPECT_TRUE( isConnectedPath( mesh.topology, start, *fm, end ) );
                const auto path = computeSurfacePath( mp, start, end, 5, vertRegions[i] );
                ASSERT_TRUE( path.has_value() );
                EXPECT_TRUE( isConnectedPath( mesh.topology, start, *path, end ) );
            }
        }
    }
}

TEST( MRMesh, ShortestEdgePathInRegion )
{
    // a planar grid 8x8, the region of cells is U-shaped: two arms separated by the gap column 4 and joined by the bottom row;
    // the edges across the gap have both ends in the region vertices, but no region face
    const int n = 8;
    Mesh mesh;
    for ( int y = 0; y <= n; ++y )
        for ( int x = 0; x <= n; ++x )
            mesh.points.emplace_back( float( x ), float( y ), 0.f );
    Triangulation t;
    FaceBitSet faces;
    for ( int y = 0; y < n; ++y )
    {
        for ( int x = 0; x < n; ++x )
        {
            const VertId a( y * ( n + 1 ) + x ), b( a + 1 ), c( a + n + 1 ), d( c + 1 );
            t.push_back( { a, b, d } );
            t.push_back( { a, d, c } );
            const bool inRegion = x != 4 || y == 0;
            faces.autoResizeSet( FaceId( int( t.size() ) - 2 ), inRegion );
            faces.autoResizeSet( FaceId( int( t.size() ) - 1 ), inRegion );
        }
    }
    mesh.topology = MeshBuilder::fromTriangles( t );
    const auto verts = getIncidentVerts( mesh.topology, faces );
    const MeshTriPoint start( mesh.topology, VertId( 7 * ( n + 1 ) + 1 ) ), end( mesh.topology, VertId( 7 * ( n + 1 ) + 7 ) );

    // the length of the path and whether every its step passes an edge with a region face on a side
    auto measure = [&]( const SurfacePath & path, float & len )
    {
        std::vector<VertId> vs{ start.inVertex( mesh.topology ) };
        for ( const auto & ep : path )
        {
            const auto v = ep.inVertex( mesh.topology );
            EXPECT_TRUE( v );
            if ( v != vs.back() )
                vs.push_back( v );
        }
        if ( vs.back() != end.inVertex( mesh.topology ) )
            vs.push_back( end.inVertex( mesh.topology ) );
        len = 0;
        bool byFaces = true;
        for ( size_t i = 0; i + 1 < vs.size(); ++i )
        {
            const auto e = mesh.topology.findEdge( vs[i], vs[i + 1] );
            EXPECT_TRUE( e );
            if ( !e )
                return false;
            len += ( mesh.points[vs[i + 1]] - mesh.points[vs[i]] ).length();
            const auto l = mesh.topology.left( e ), r = mesh.topology.right( e );
            byFaces = byFaces && ( ( l && faces.test( l ) ) || ( r && faces.test( r ) ) );
        }
        return byFaces;
    };

    const auto byFaces = computeShortestEdgePath( { mesh, &faces }, start, end );
    ASSERT_TRUE( byFaces.has_value() );
    EXPECT_TRUE( isConnectedPath( mesh.topology, start, *byFaces, end ) );
    float lenFaces = 0;
    EXPECT_TRUE( measure( *byFaces, lenFaces ) );

    const auto byVerts = computeShortestEdgePath( mesh, start, end, &verts );
    ASSERT_TRUE( byVerts.has_value() );
    EXPECT_TRUE( isConnectedPath( mesh.topology, start, *byVerts, end ) );
    float lenVerts = 0;
    EXPECT_FALSE( measure( *byVerts, lenVerts ) ); // crosses the gap
    for ( const auto & ep : *byVerts )
        EXPECT_TRUE( verts.test( ep.inVertex( mesh.topology ) ) );

    // the face region forces the path around the gap through the bottom row
    EXPECT_GT( lenFaces, lenVerts + 10 );
}

} //namespace MR
