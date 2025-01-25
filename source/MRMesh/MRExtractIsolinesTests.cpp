#include "MRGTest.h"
#include "MRExtractIsolines.h"
#include "MRMesh.h"
#include "MRObjectMesh.h"
#include "MRCube.h"


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
    Mesh mesh = MR::makeCube( Vector3f::diagonal( 1.F ), Vector3f() );

    auto res = extractXYPlaneSections( mesh, -0.5f );
    EXPECT_EQ( res.size(), 0 );

    const auto testLevel = 0.5f;
    res = extractXYPlaneSections( mesh, testLevel );
    EXPECT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 9 );
    EXPECT_EQ( res[0].front(), res[0].back() );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LT( std::abs( point.z - testLevel ), 1e-6f );
    }
    EXPECT_EQ( findTriangleSectionsByXYPlane( mesh, testLevel ).size(), 8 );

    FaceBitSet fs;
    fs.autoResizeSet( 5_f );
    fs.set( 2_f );
    res = extractXYPlaneSections( { mesh, &fs }, testLevel );
    EXPECT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 3 );
    EXPECT_NE( res[0].front(), res[0].back() );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LT( std::abs( point.z - testLevel ), 1e-6f );
    }
    EXPECT_EQ( findTriangleSectionsByXYPlane( { mesh, &fs }, testLevel ).size(), 2 );

    // make a hole in mesh to extract not closed contour
    mesh.deleteFaces( fs );
    res = extractXYPlaneSections( mesh, testLevel );
    EXPECT_EQ( res.size(), 1 );
    EXPECT_EQ( res[0].size(), 7 );
    EXPECT_NE( res[0].front(), res[0].back() );
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LT( std::abs( point.z - testLevel ), 1e-6f );
    }
    EXPECT_EQ( findTriangleSectionsByXYPlane( mesh, testLevel ).size(), 6 );
}

TEST( MRMesh, TrackPlaneSection )
{
    Mesh mesh = MR::makeCube( Vector3f::diagonal( 1.F ), Vector3f::diagonal( -0.5F ) );
    const MeshTriPoint start{ 10_e, { 0.25f, 0.25f } };
    const MeshTriPoint end{ 21_e, { 0.25f, 0.25f } };
    const Vector3f planePoint{ -0.5f, -0.5f, 0.0f };
    auto res = trackSection( mesh, start, end, planePoint, true );
    EXPECT_TRUE( res.has_value() );
    EXPECT_EQ( res->size(), 1 );
    EXPECT_EQ( (*res)[0].e, 11_e );
    EXPECT_LT( std::abs( (*res)[0].a - 0.5f ), 1e-6f );
}

} // namespace MR
