#include <MRMesh/MRPointToPlaneAligningTransform.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRRigidScaleXf3.h>
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, PointToPlaneAligningTransform1 )
{
    std::vector<Vector3d> pInit, n, n2;
    pInit.resize( 10 );
    n.resize( 10 );
    n2.resize( 3 );

    pInit[0]  = {   1.0,   1.0, -5.0 }; n[0] = {  0.0,  0.0, -1.0 }; n2[0] = { 0.1, -0.1,  0.0 };
    pInit[1]  = {  14.0,   1.0,  1.0 }; n[1] = {  1.0,  0.1,  1.0 }; n2[1] = { 0.3,  0.0, -0.3 };
    pInit[2]  = {   1.0,  14.0,  2.0 }; n[2] = {  0.1,  1.0,  1.2 }; n2[2] = { 0.0, -0.6,  0.5 };
    pInit[3]  = { -11.0,   2.0,  3.0 }; n[3] = { -1.0,  0.1,  1.0 };
    pInit[4]  = {   1.0, -11.0,  4.0 }; n[4] = {  0.1, -1.1,  1.1 };
    pInit[5]  = {   1.0,   2.0,  8.0 }; n[5] = {  0.1,  0.1,  1.0 };
    pInit[6]  = {   2.0,   1.0, -5.0 }; n[6] = {  0.1,  0.0, -1.0 };
    pInit[7]  = {  15.0,   1.5,  1.0 }; n[7] = {  1.1,  0.1,  1.0 };
    pInit[8]  = {   1.5,  15.0,  2.0 }; n[8] = {  0.1,  1.0,  1.2 };
    pInit[9]  = { -11.0,   2.5,  3.1 }; n[9] = { -1.1,  0.1,  1.1 };

    auto preparePt2Pl = [&]( const AffineXf3d & xf )
    {
        std::vector<Vector3d> pTransformed( 10 );

        for( int i = 0; i < 10; i++ )
            pTransformed[i] = xf( pInit[i] );
        for( int i = 0; i < 3; i++ )
            pTransformed[i] += n2[i];

        PointToPlaneAligningTransform p2pl;
        for( int i = 0; i < 10; i++ )
            p2pl.add( pInit[i], pTransformed[i], n[i] );
        p2pl.prepare();
        return p2pl;
    };

    double alpha = 0.15, beta = 0.23, gamma = -0.17;
    const Vector3d eulerAngles{ alpha, beta, gamma };
    const auto [e1, e2] = eulerAngles.perpendicular();
    Matrix3d rotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( eulerAngles );
    const Vector3d b( 2., 3., -1. );
    AffineXf3d xf1( rotationMatrix, b );

    const auto ptp1 = preparePt2Pl( xf1 );
    constexpr double eps = 3e-13;

    {
        const auto ammendment = ptp1.calculateAmendment();
        EXPECT_EQ( ammendment.s, 1 );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
        auto shift = ptp1.findBestTranslation( ammendment.a, ammendment.s );
        EXPECT_NEAR( ( xf1.b - shift ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateAmendmentWithScale();
        EXPECT_NEAR( ammendment.s, 1., 1e-13 );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
        auto shift = ptp1.findBestTranslation( ammendment.a, ammendment.s );
        EXPECT_NEAR( ( xf1.b - shift ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateFixedAxisAmendment( 10.0 * eulerAngles );
        EXPECT_EQ( ammendment.s, 1 );
        EXPECT_NEAR( cross( ammendment.a, eulerAngles.normalized() ).length(), 0., eps );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
        auto shift = ptp1.findBestTranslation( ammendment.a, ammendment.s );
        EXPECT_NEAR( ( xf1.b - shift ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateFixedAxisAmendment( e1 );
        EXPECT_EQ( ammendment.s, 1 );
        EXPECT_NEAR( cross( ammendment.a, e1 ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateOrthogonalAxisAmendment( -12.0 * e1 );
        EXPECT_EQ( ammendment.s, 1 );
        EXPECT_NEAR( dot( ammendment.a, e1 ), 0., eps );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
        auto shift = ptp1.findBestTranslation( ammendment.a, ammendment.s );
        EXPECT_NEAR( ( xf1.b - shift ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateOrthogonalAxisAmendment( 12.0 * e2 );
        EXPECT_EQ( ammendment.s, 1 );
        EXPECT_NEAR( dot( ammendment.a, e2 ), 0., eps );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
        auto shift = ptp1.findBestTranslation( ammendment.a, ammendment.s );
        EXPECT_NEAR( ( xf1.b - shift ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateOrthogonalAxisAmendment( eulerAngles );
        EXPECT_EQ( ammendment.s, 1 );
        EXPECT_NEAR( dot( ammendment.a, eulerAngles.normalized() ), 0., eps );
    }

    {
        const double scale = 0.5;
        AffineXf3d xf2( scale * rotationMatrix, b );
        const auto ptp2 = preparePt2Pl( xf2 );
        const auto ammendment = ptp2.calculateAmendmentWithScale();
        EXPECT_NEAR( ammendment.s, scale, 1e-13 );
        auto xf3 = ammendment.linearXf();
        EXPECT_NEAR( ( xf3.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf3.b - xf2.b ).length(), 0., eps );
        auto shift = ptp2.findBestTranslation( ammendment.a, ammendment.s );
        EXPECT_NEAR( ( xf2.b - shift ).length(), 0., eps );
    }

    {
        AffineXf3d xf( {}, b );
        const auto p2pl = preparePt2Pl( xf );
        const auto shift = p2pl.findBestTranslation();
        EXPECT_NEAR( ( b - shift ).length(), 0., eps );
    }
}

TEST( MRMesh, PointToPlaneAligningTransform2 )
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

    // large absolute value testing
    //AffineXf3d largeShiftXf = AffineXf3d(Matrix3d(), Vector3d(/*10000, - 10000, 0*/));
    //for (auto& pn : points) pn = largeShiftXf(pn);

    const std::vector<Vector3d> pointsNorm = {
        Vector3d{  0.0,  0.0, -1.0 }.normalized(),
        Vector3d{  1.0,  0.1,  1.0 }.normalized(),
        Vector3d{  0.1,  1.0,  1.2 }.normalized(),
        Vector3d{ -1.0,  0.1,  1.0 }.normalized(),
        Vector3d{  0.1, -1.1,  1.1 }.normalized(),
        Vector3d{  0.1,  0.1,  1.0 }.normalized(),
        Vector3d{  0.1,  0.0, -1.0 }.normalized(),
        Vector3d{  1.1,  0.1,  1.0 }.normalized(),
        Vector3d{  0.1,  1.0,  1.2 }.normalized(),
        Vector3d{ -1.1,  0.1,  1.1 }.normalized()
    };

    // init translation
    const std::vector<AffineXf3d> xfs = {
        // zero xf
        AffineXf3d(
            Matrix3d(
                Vector3d(1, 0, 0),
                Vector3d(0, 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // Rz
        AffineXf3d(
            Matrix3d(
                Vector3d(1, sin(0.5), 0),
                Vector3d(-sin(0.5), 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // Rz + transl
        AffineXf3d(
            Matrix3d(
                Vector3d(1, sin(0.5), 0),
                Vector3d(-sin(0.5), 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(2,-2,0)),

        // complex xf
        AffineXf3d(
            Matrix3d(
                Vector3d(1, sin(0.15), -sin(0.1)),
                Vector3d(-sin(0.15), 1, sin(0.2)),
                Vector3d(sin(0.1), -sin(0.2), 1)
            ),
            Vector3d(2,-20,8)),
    };

    //std::random_device rd;
    //std::mt19937 gen(rd());
    //const double max_rnd = 0.01;
    //std::uniform_real_distribution<> dis(-max_rnd, max_rnd);
    for (const auto& xf : xfs)
    {
        constexpr double eps = 5e-13;
        {
            PointToPlaneAligningTransform p2pl;
            for (int i = 0; i < points.size(); i++)
            {
                p2pl.add( points[i],
                    xf( points[i] ), // +Vector3d(dis(gen), dis(gen), dis(gen))
                    xf.A * pointsNorm[i] );
            }
            p2pl.prepare();
            auto am = p2pl.calculateAmendment();
            AffineXf3d xfResP2pl = am.linearXf();

            EXPECT_NEAR((xfResP2pl.A - xf.A).norm(), 0., eps);
            EXPECT_NEAR((xfResP2pl.b - xf.b).length(), 0., eps);

            auto shift = p2pl.findBestTranslation( am.a, am.s );
            EXPECT_NEAR( ( xf.b - shift ).length(), 0., eps );
        }
        {
            auto scaleXf = xf;
            scaleXf.A *= 0.3;
            PointToPlaneAligningTransform p2pl;
            for (int i = 0; i < points.size(); i++)
            {
                p2pl.add( points[i],
                    scaleXf( points[i] ), // +Vector3d(dis(gen), dis(gen), dis(gen))
                    xf.A * pointsNorm[i] );
            }
            p2pl.prepare();
            auto am = p2pl.calculateAmendmentWithScale();
            AffineXf3d xfResP2pl = am.linearXf();

            EXPECT_NEAR((xfResP2pl.A - scaleXf.A).norm(), 0., eps);
            EXPECT_NEAR((xfResP2pl.b - scaleXf.b).length(), 0., eps);

            auto shift = p2pl.findBestTranslation( am.a, am.s );
            EXPECT_NEAR( ( xf.b - shift ).length(), 0., eps );
        }
    }
}

} //namespace MR
