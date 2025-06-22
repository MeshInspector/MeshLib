#include <MRMesh/MRGTest.h>
#include <MRMesh/MRExtractIsolines.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRLineSegm.h>

namespace MR
{

TEST( MRMesh, ExtractPlaneSections )
{
    Mesh mesh = MR::makeCube( Vector3f::diagonal( 1.F ), Vector3f() );
    Plane3f plane = Plane3f{ Vector3f::diagonal( 1.F ), 1.F }.normalized();

    constexpr float delta = std::numeric_limits<float>::epsilon() * 10.F;

    PlaneSections res = extractPlaneSections( mesh, plane );
    EXPECT_EQ( res.size(), 1 );

    plane.d = 0.F - delta;
    res = extractPlaneSections( mesh, plane );
    EXPECT_EQ( res.size(), 0 );

    plane.d = 0.F + delta;
    res = extractPlaneSections( mesh, plane );
    EXPECT_EQ( res.size(), 1 );

    plane.d = sqrt( 3.F ) - delta;
    res = extractPlaneSections( mesh, plane );
    EXPECT_EQ( res.size(), 1 );

    plane.d = sqrt( 3.F ) + delta;
    res = extractPlaneSections( mesh, plane );
    EXPECT_EQ( res.size(), 0 );

    
    plane.n = Vector3f::plusX();
    plane.d = 0.4F;
    res = extractPlaneSections( mesh, plane );
    ASSERT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 9 );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LE( std::abs( plane.distance( point ) ), delta );
    }


    plane.n = Vector3f::diagonal( 1.F ).normalized();
    plane.d = sqrt( 3.F ) / 2.F;
    res = extractPlaneSections( mesh, plane );
    ASSERT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 13 );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LE( std::abs( plane.distance( point ) ), delta );
    }


    plane.n = Vector3f{ 1.F, 2.F, 3.F }.normalized();
    plane.d = 0.646F;
    res = extractPlaneSections( mesh, plane );
    ASSERT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 11 );
    EXPECT_EQ( res[0].front(), res[0].back() );
    EXPECT_EQ( getCrossedFaces( mesh.topology, res[0] ).count(), 10 );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LE( std::abs( plane.distance( point ) ), delta );
    }

    // make a hole in mesh to extract not closed contour
    FaceBitSet fs;
    fs.autoResizeSet( 2_f );
    mesh.deleteFaces( fs );
    res = extractPlaneSections( mesh, plane );
    ASSERT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 10 );
    EXPECT_NE( res[0].front(), res[0].back() );
    EXPECT_EQ( getCrossedFaces( mesh.topology, res[0] ).count(), 9 );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LE( std::abs( plane.distance( point ) ), delta );
    }
}

TEST( MRMesh, ExtractXYPlaneSections )
{
    auto testSection = []( const MeshPart & mp, float zLevel )
    {
        auto res = extractXYPlaneSections( mp, zLevel, UseAABBTree::No );
        EXPECT_EQ( res, extractXYPlaneSections( mp, zLevel, UseAABBTree::Yes ) );
        bool has = !res.empty();
        EXPECT_EQ( has, hasAnyXYPlaneSection( mp, zLevel, UseAABBTree::No ) );
        EXPECT_EQ( has, hasAnyXYPlaneSection( mp, zLevel, UseAABBTree::Yes ) );

        Plane3f plane( Vector3f( 0, 0, 1 ), zLevel );
        EXPECT_EQ( res, extractPlaneSections( mp, plane, UseAABBTree::No ) );
        EXPECT_EQ( res, extractPlaneSections( mp, plane, UseAABBTree::Yes ) );
        EXPECT_EQ( has, hasAnyPlaneSection( mp, plane, UseAABBTree::No ) );
        EXPECT_EQ( has, hasAnyPlaneSection( mp, plane, UseAABBTree::Yes ) );
        return res;
    };

    Mesh mesh = MR::makeCube( Vector3f::diagonal( 1.F ), Vector3f() );

    auto res = testSection( mesh, -0.5f );
    EXPECT_EQ( res.size(), 0 );

    const auto testLevel = 0.5f;
    res = testSection( mesh, testLevel );
    EXPECT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 9 );
    EXPECT_EQ( res[0].front(), res[0].back() );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LT( std::abs( point.z - testLevel ), 1e-6f );
    }
    EXPECT_EQ( findTriangleSectionsByXYPlane( mesh, testLevel, nullptr, UseAABBTree::No ).size(), 8 );
    EXPECT_EQ( findTriangleSectionsByXYPlane( mesh, testLevel, nullptr, UseAABBTree::Yes ).size(), 8 );

    FaceBitSet fs;
    fs.autoResizeSet( 5_f );
    fs.set( 2_f );
    res = testSection( { mesh, &fs }, testLevel );
    EXPECT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 3 );
    EXPECT_NE( res[0].front(), res[0].back() );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LT( std::abs( point.z - testLevel ), 1e-6f );
    }
    EXPECT_EQ( findTriangleSectionsByXYPlane( { mesh, &fs }, testLevel, nullptr, UseAABBTree::No ).size(), 2 );
    EXPECT_EQ( findTriangleSectionsByXYPlane( { mesh, &fs }, testLevel, nullptr, UseAABBTree::Yes ).size(), 2 );

    // make a hole in mesh to extract not closed contour
    mesh.deleteFaces( fs );
    res = testSection( mesh, testLevel );
    EXPECT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 7 );
    EXPECT_NE( res[0].front(), res[0].back() );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LT( std::abs( point.z - testLevel ), 1e-6f );
    }
    EXPECT_EQ( findTriangleSectionsByXYPlane( mesh, testLevel, nullptr, UseAABBTree::No ).size(), 6 );
    EXPECT_EQ( findTriangleSectionsByXYPlane( mesh, testLevel, nullptr, UseAABBTree::Yes ).size(), 6 );
}

TEST( MRMesh, TrackPlaneSection )
{
    const Mesh mesh = MR::makeCube( Vector3f::diagonal( 1.F ), Vector3f::diagonal( -0.5F ) );
    const float eps = 1e-6f;

    // central plane, from triangle centroid to triangle centroid, ccw direction
    const MeshTriPoint start{ 10_e, { 0.25f, 0.25f } };
    const MeshTriPoint end{ 21_e, { 0.25f, 0.25f } };
    const Vector3f planePoint{ -0.5f, -0.5f, 0.0f };
    auto res = trackSection( mesh, start, end, planePoint, true );
    EXPECT_TRUE( res.has_value() );
    EXPECT_EQ( res->size(), 1 );
    EXPECT_EQ( (*res)[0].e, 11_e );
    EXPECT_LT( std::abs( (*res)[0].a - 0.5f ), eps );

    // central plane, from triangle centroid to triangle centroid, cw direction
    res = trackSection( mesh, start, end, planePoint, false );
    EXPECT_TRUE( res.has_value() );
    EXPECT_EQ( res->size(), 7 );
    for ( const auto & p : *res )
        EXPECT_LT( std::abs( p.a - 0.5f ), eps );

    // central plane, from edge centroid to triangle centroid, ccw direction
    const MeshTriPoint startE{ 10_e, { 0.5f, 0.0f } };
    res = trackSection( mesh, startE, end, planePoint, true );
    EXPECT_TRUE( res.has_value() );
    EXPECT_EQ( res->size(), 0 );

    // central plane, from edge centroid to triangle centroid, cc direction
    res = trackSection( mesh, startE, end, planePoint, false );
    EXPECT_TRUE( res.has_value() );
    // EXPECT_EQ( res->size(), 8 ); //broken currently
    for ( const auto & p : *res )
        EXPECT_LT( std::abs( p.a - 0.5f ), eps );
}

TEST( MRMesh, TrackPlaneSectionOnDistance )
{
    Mesh mesh = MR::makeCube( Vector3f::diagonal( 1.F ), Vector3f::diagonal( -0.5F ) );
    const float eps = 1e-6f;

    const MeshTriPoint start{ 10_e, { 0.25f, 0.25f } };
    MeshTriPoint finish;

    // track within same triangle, one direction
    auto sec = trackSection( mesh, start, finish, Vector3f( 0, 1, 0 ), 0.1f );
    EXPECT_EQ( sec.size(), 0 );
    EXPECT_EQ( finish.e, 10_e );
    EXPECT_LT( std::abs( distance( mesh.triPoint( start ), mesh.triPoint( finish ) ) - 0.1f ), eps );

    // track within same triangle, another direction
    finish = {};
    sec = trackSection( mesh, start, finish, Vector3f( 0, -1, 0 ), 0.1f );
    EXPECT_EQ( sec.size(), 0 );
    EXPECT_EQ( finish.e, 10_e );
    EXPECT_LT( std::abs( distance( mesh.triPoint( start ), mesh.triPoint( finish ) ) - 0.1f ), eps );

    // track to the next triangle in same plane
    finish = {};
    sec = trackSection( mesh, start, finish, Vector3f( 0, 1, 0 ), 0.5f );
    EXPECT_EQ( sec.size(), 1 );
    EXPECT_EQ( sec[0].e, 15_e );
    EXPECT_LT( std::abs( sec[0].a - 0.5f ), eps );
    EXPECT_EQ( mesh.topology.left( finish.e ), 3_f );
    EXPECT_LT( std::abs( distance( mesh.triPoint( start ), mesh.triPoint( finish ) ) - 0.5f ), eps );

    // ...one triangle further
    finish = {};
    sec = trackSection( mesh, start, finish, Vector3f( 0, 1, 0 ), 1.0f );
    EXPECT_EQ( sec.size(), 2 );
    EXPECT_EQ( sec[0].e, 15_e );
    EXPECT_LT( std::abs( sec[0].a - 0.5f ), eps );
    EXPECT_EQ( sec[1].e, 17_e );
    EXPECT_LT( std::abs( sec[1].a - 0.5f ), eps );
    EXPECT_EQ( mesh.topology.left( finish.e ), 8_f );
    EXPECT_LT( distance( Vector3f( -0.25f, 0.5f, 0.0f ), mesh.triPoint( finish ) ), eps );

    // track to the next triangle in the opposite direction
    finish = {};
    sec = trackSection( mesh, start, finish, Vector3f( 0, -1, 0 ), 0.5f );
    EXPECT_EQ( sec.size(), 1 );
    EXPECT_EQ( sec[0].e, 11_e );
    EXPECT_LT( std::abs( sec[0].a - 0.5f ), eps );
    EXPECT_EQ( mesh.topology.left( finish.e ), 5_f );
    EXPECT_LT( distance( mesh.triPoint( finish ), Vector3f( -0.25f, -0.5f, 0.0f ) ), eps );

    // track from edge's center
    const MeshTriPoint startE{ 10_e, { 0.5f, 0.0f } };
    finish = {};
    sec = trackSection( mesh, startE, finish, Vector3f( -1, 1, 0 ).normalized(), 0.25f );
    EXPECT_EQ( sec.size(), 0 );
    EXPECT_EQ( mesh.topology.left( finish.e ), 2_f );
    EXPECT_LT( distance( mesh.triPoint( finish ), Vector3f( -0.5f, -0.25f, 0.0f ) ), eps );

    // track from edge's center in the opposite direction
    finish = {};
    sec = trackSection( mesh, startE, finish, Vector3f( 1, -1, 0 ).normalized(), 0.25f );
    EXPECT_EQ( sec.size(), 0 );
    EXPECT_EQ( mesh.topology.left( finish.e ), 5_f );
    EXPECT_LT( distance( mesh.triPoint( finish ), Vector3f( -0.25f, -0.5f, 0.0f ) ), eps );

    // track from vertex
    const MeshTriPoint startV{ 10_e, { 0.0f, 0.0f } };
    finish = {};
    sec = trackSection( mesh, startV, finish, Vector3f( 1, 0, 2 ).normalized(), 1.0f );
    EXPECT_EQ( sec.size(), 0 );
    EXPECT_EQ( mesh.topology.left( finish.e ), 5_f );
    EXPECT_LT( std::abs( distance( mesh.triPoint( startV ), mesh.triPoint( finish ) ) - 1.0f ), eps );

    // track from vertex, another direction
    finish = {};
    sec = trackSection( mesh, startV, finish, Vector3f( 2, 0, 1 ).normalized(), 1.0f );
    EXPECT_EQ( sec.size(), 0 );
    EXPECT_EQ( mesh.topology.left( finish.e ), 4_f );
    EXPECT_LT( std::abs( distance( mesh.triPoint( startV ), mesh.triPoint( finish ) ) - 1.0f ), eps );

    // track from another vertex, with one edge crossing
    const MeshTriPoint startV2{ 10_e, { 1.0f, 0.0f } };
    finish = {};
    sec = trackSection( mesh, startV2, finish, Vector3f( 1, 0, -1 ).normalized(), 1.0f );
    EXPECT_EQ( sec.size(), 1 );
    EXPECT_EQ( sec[0].e, 20_e );
    EXPECT_LT( std::abs( sec[0].a - 0.5f ), eps );
    EXPECT_EQ( mesh.topology.left( finish.e ), 4_f );
    EXPECT_LT( std::abs( distance( mesh.triPoint( startV2 ), mesh.triPoint( finish ) ) - 1.0f ), eps );

    // delete triangle on track path, expect finish on hole's edge
    FaceBitSet fs( 9, false );
    fs.set( 8_f );
    mesh.deleteFaces( fs );
    finish = {};
    sec = trackSection( mesh, start, finish, Vector3f( 0, 1, 0 ), 1.0f );
    EXPECT_EQ( sec.size(), 1 );
    EXPECT_EQ( sec[0].e, 15_e );
    EXPECT_LT( std::abs( sec[0].a - 0.5f ), eps );
    EXPECT_EQ( mesh.topology.left( finish.e ), FaceId{} );
    auto finishE = finish.onEdge( mesh.topology );
    EXPECT_EQ( finishE.e, 17_e );
    EXPECT_LT( std::abs( finishE.a - 0.5f ), eps );
}

} // namespace MR
