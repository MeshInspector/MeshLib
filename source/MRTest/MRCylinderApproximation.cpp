#include "MRMesh/MRCylinder3.h"
#include "MRMesh/MRCylinderApproximator.h"

#include "MRMesh/MRGTest.h"

namespace
{

constexpr int phiResolution = 180;
constexpr int thetaResolution = 180;

} // namespace

namespace MR
{

TEST( MRMesh, CylinderApproximation )
{
    float originalRadius = 1.5f;
    float originalLength = 10.0f;
    float startAngle = 0.0f;
    float archSize = PI_F / 1.5f;
    int  resolution = 100;

    AffineXf3f testXf;
    Vector3f center{ 1,2,3 };
    Vector3f direction = ( Vector3f{ 3,2,1 } ).normalized();

    testXf = AffineXf3f::translation( { 1,2,3 } );
    testXf.A = Matrix3f::rotation( Vector3f::plusZ(), direction ) * Matrix3f::scale( { originalRadius , originalRadius  ,originalLength } );

    std::vector<Vector3f> points;

    float angleStep = archSize / resolution;
    float zStep = 1.0f / resolution;
    for ( int i = 0; i < resolution; ++i )
    {
        float angle = startAngle + i * angleStep;
        float z = i * zStep - 0.5f;
        points.emplace_back( testXf( Vector3f{ cosf( angle )  , sinf( angle ) , z } ) );
        points.emplace_back( testXf( Vector3f{ cosf( angle )  , sinf( angle ) ,  -z } ) );
    }

    /////////////////////////////
    // General multithread test
    /////////////////////////////

    Cylinder3<float> result;
    auto fit = Cylinder3Approximation<float>();
    auto approximationRMS = fit.solveGeneral( points, result, phiResolution, thetaResolution, true );
    std::cout << "multi thread center: " << result.center() << " direction:" << result.direction() << " length:" << result.length << " radius:" << result.radius << " error:" << approximationRMS << std::endl;

    EXPECT_LE( approximationRMS, 0.1f );
    EXPECT_NEAR( result.radius, originalRadius, 0.1f );
    EXPECT_NEAR( result.length, originalLength, 0.1f );
    EXPECT_LE( ( result.center() - center ).length(), 0.1f );
    EXPECT_GT( dot( direction, result.direction() ), 0.9f );

    ///////////////////////////////////////
    // Compare single thread vs multithread
    ///////////////////////////////////////

    Cylinder3<float> resultST;
    auto approximationRMS_ST = fit.solveGeneral( points, resultST, phiResolution, thetaResolution, false );
    std::cout << "single thread center: " << result.center() << " direction:" << result.direction() << " length:" << result.length << " radius:" << result.radius << " error:" << approximationRMS << std::endl;

    EXPECT_NEAR( approximationRMS, approximationRMS_ST, 0.01f );
    EXPECT_NEAR( result.radius, resultST.radius, 0.01f );
    EXPECT_NEAR( result.length, resultST.length, 0.01f );
    EXPECT_LE( ( result.center() - resultST.center() ).length(), 0.01f );
    EXPECT_GT( dot( resultST.direction(), result.direction() ), 0.99f );

    //////////////////////////////////////////
    // Test usage with SpecificAxisFit (SAF)
    //////////////////////////////////////////

    Cylinder3<float> resultSAF;
    Vector3f noice{ 0.002f , -0.003f , 0.01f };

    auto approximationRMS_SAF = fit.solveSpecificAxis( points, resultSAF, direction + noice );
    std::cout << "SpecificAxisFit center: " << resultSAF.center() << " direction:" << resultSAF.direction() << " length:" << resultSAF.length << " radius:" << resultSAF.radius << " error:" << approximationRMS_SAF << std::endl;

    EXPECT_LE( approximationRMS_SAF, 0.1f );
    EXPECT_NEAR( resultSAF.radius, originalRadius, 0.1f );
    EXPECT_NEAR( resultSAF.length, originalLength, 0.1f );
    EXPECT_LE( ( resultSAF.center() - center ).length(), 0.1f );
    EXPECT_GT( dot( direction, resultSAF.direction() ), 0.9f );
}

} // namespace MR
