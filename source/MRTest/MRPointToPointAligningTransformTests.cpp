#include <MRMesh/MRPointToPointAligningTransform.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, PointToPointAligningTransform1 )
{
    PointToPointAligningTransform at1;

    Vector3d b( 1, 1, 1 );
    at1.add( Vector3d::plusX(), Vector3d::plusY() + b );
    at1.add( Vector3d::plusY(), Vector3d::plusZ() + b );
    at1.add( Vector3d::plusZ(), Vector3d::plusX() + b );

    auto xf1 = at1.findBestRigidXf();
    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    xf1 = at1.findBestRigidScaleXf();
    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    xf1 = at1.findBestRigidXfFixedRotationAxis( Vector3d{ 1, 1, 1 } );
    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    xf1 = at1.findBestRigidXfOrthogonalRotationAxis( Vector3d{1, 0, -1} );
    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    // Second test
    PointToPointAligningTransform at2;
    Vector3d p11, p12, p21, p22, p31, p32, p41, p42, p51, p52;
    p11 = { 2, 1, 1 }; p12 = { -7, -5, -1 };
    p21 = { 4, 1, 1 }; p22 = { -7, -7, -1 };
    p31 = { 4, 4, 1 }; p32 = { -7, -7, -4 };
    p41 = { 4, 4, 6 }; p42 = { -2, -7, -4 };
    p51 = { 3, 4, 6 }; p52 = { -2, -6, -4 };
    at2.add( p11, p12 );
    at2.add( p21, p22 );
    at2.add( p31, p32 );
    at2.add( p41, p42 );
    at2.add( p51, p52 );

    auto xf2 = at2.findBestRigidXf();
    EXPECT_NEAR( ( xf2( p11 ) - p12 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p21 ) - p22 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p31 ) - p32 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p41 ) - p42 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p51 ) - p52 ).length(), 0., 1e-14 );

    auto xf2_ = at2.findBestRigidXfFixedRotationAxis( Vector3d{ -1, 1, -1 } );
    EXPECT_NEAR( ( xf2_( p11 ) - p12 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p21 ) - p22 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p31 ) - p32 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p41 ) - p42 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p51 ) - p52 ).length(), 0., 1e-14 );

    // Third test
    PointToPointAligningTransform at3;
    p11 = { 10.0, 10.0, 20.0 }; p12 = { 16.0, 23.0, 10.0 };
    p21 = { 3.1, 4.3, 5.2 }; p22 = { 10.3, 8.2, 3.1 };
    p31 = { 0.1, 0.1, -0.1 }; p32 = { 6.1, 2.9, 0.1 };
    p41 = { -0.5, 4.2, 6.1 }; p42 = { 10.2, 9.1, -0.5 };
    p51 = { 3.8, 4.1, -1.5 }; p52 = { 10.1, 1.5, 3.8 };
    at3.add( p11, p12 );
    at3.add( p21, p22 );
    at3.add( p31, p32 );
    at3.add( p41, p42 );
    at3.add( p51, p52 );

    auto xf3 = at3.findBestRigidXf();
    EXPECT_NEAR( ( xf3( p11 ) - p12 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p21 ) - p22 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p31 ) - p32 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p41 ) - p42 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p51 ) - p52 ).length(), 0., 1e-13 );

    auto xf3_ = at3.findBestRigidXfFixedRotationAxis( Vector3d{ 1, 1, 1 } );
    EXPECT_NEAR( ( xf3_( p11 ) - p12 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p21 ) - p22 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p31 ) - p32 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p41 ) - p42 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p51 ) - p52 ).length(), 0., 1e-13 );


    PointToPointAligningTransform at4;
    double scale = 0.25;
    at4.add( Vector3d::plusX(), scale * Vector3d::plusY() + b );
    at4.add( Vector3d::plusY(), scale * Vector3d::plusZ() + b );
    at4.add( Vector3d::plusZ(), scale * Vector3d::plusX() + b );

    auto xf4 = at4.findBestRigidScaleXf();
    ASSERT_NEAR( ( xf4( Vector3d::plusX() ) - scale * Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf4( Vector3d::plusY() ) - scale * Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf4( Vector3d::plusZ() ) - scale * Vector3d::plusX() - b ).length(), 0., 1e-15 );
}

TEST(MRMesh, PointToPointAligningTransform2 )
{
    // set points
    const std::vector<Vector3d> points = {
        {   1.0,   1.0, -5.0 },
        {  14.0,   1.0,  1.0 },
        {   1.0,  14.0,  2.0 },
        { -11.0,   2.0,  3.0 },
        {   1.0, -11.0,  4.0 },
        {   1.0,   2.0,  8.0 },
        {   2.0,   1.0, -5.0 },
        {  15.0,   1.5,  1.0 },
        {   1.5,  15.0,  2.0 },
        { -11.0,   2.5,  3.1 },
    };
    // Point to Point part
    const std::vector<AffineXf3d> xfs = {
        // zero xf
        AffineXf3d(
            Matrix3d(
                Vector3d(1, 0, 0),
                Vector3d(0, 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // small Rz
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0.6, 0),
                Vector3d(-0.6, 0.8, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // small transl
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0.6, 0),
                Vector3d(-0.6, 0.8, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(2,-2,0)),

        // complex xf
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0, -0.6),
                Vector3d(0, 1, 0),
                Vector3d(0.6, 0, 0.8)
            ),
            Vector3d(200,-200,0)),
    };

    constexpr auto eps = 5e-14;
    for ( const auto& xf : xfs )
    {
        {
            PointToPointAligningTransform p2pt;
            for ( const auto & p : points )
                p2pt.add( p, xf( p ) );

            auto xfResP2pt = p2pt.findBestRigidXf();
            EXPECT_NEAR((xfResP2pt.A - xf.A).norm(), 0., eps);
            EXPECT_NEAR((xfResP2pt.b - xf.b).length(), 0., eps);
        }
        {
            auto scaleXf = xf;
            scaleXf.A *= 3.0;

            PointToPointAligningTransform p2ptS;
            for ( const auto & p : points )
                p2ptS.add( p, scaleXf( p ) );

            auto xfResP2ptS = p2ptS.findBestRigidScaleXf();
            EXPECT_NEAR((xfResP2ptS.A - scaleXf.A).norm(), 0., eps);
            EXPECT_NEAR((xfResP2ptS.b - scaleXf.b).length(), 0., eps);
        }
    }
}

} //namespace MR
