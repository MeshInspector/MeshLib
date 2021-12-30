#include "MRGTest.h"
#include "MRExtractIsolines.h"
#include "MRMesh.h"
#include "MRObjectMesh.h"
#include "MRCube.h"


namespace MR
{

TEST( MRExtractIsolines, ExtractPlaneSectionsCube )
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
    for ( const auto& i : res[0] )
    {
        Vector3f point = mesh.edgePoint( i );
        EXPECT_LE( std::abs( plane.distance( point ) ), delta );
    }
}


} // namespace MR
