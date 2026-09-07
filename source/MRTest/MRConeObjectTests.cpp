#include <MRMesh/MRConeObject.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRConeApproximator.h>
#include <MRMesh/MRCone3.h>
#include <MRMesh/MRConstants.h>
#include <gtest/gtest.h>

namespace MR
{

namespace
{
constexpr float cBaseHeight = 1.0f;

float getNormalizedRadiusByAngle( float angle )
{
    return cBaseHeight * std::tan( angle );
}

}

TEST( MRMesh, ConeApproximation )
{
    float coneHeight = 10.0f;
    float startAngle = 0.0f;
    float archSize = PI_F / 1.5f;
    int  resolution = 100;
    float coneAngle = 12.0 * PI_F / 180.0; // 12 degree
    float noiseMaxV = 1e-3f;
    auto radius = getNormalizedRadiusByAngle( coneAngle );

    Vector3f coneApex{ 1,2,3 };
    Vector3f direction = ( Vector3f{ 3,2,1 } ).normalized();

    AffineXf3f testXf;
    testXf = AffineXf3f::translation( coneApex );
    testXf.A = Matrix3f::rotation( Vector3f::plusZ(), direction ) * Matrix3f::scale( { radius * coneHeight, radius * coneHeight, coneHeight } );

    std::vector<Vector3f> points;

    float angleStep = archSize / resolution;
    float zStep = 1.0f / resolution;
    for ( int i = 0; i < resolution; ++i )
    {
        float angle = startAngle + i * angleStep;
        float z = i * zStep;
        float radius1 = cos( coneAngle ) * z;
        float radius2 = cos( coneAngle ) * ( 1.0f - z );
        float noise = sin( z ) * noiseMaxV;   // use noise = 0 in case of test falling for more obviouse results.
        points.emplace_back( testXf( Vector3f{ cosf( angle ) * radius1 + noise , sinf( angle ) * radius1 - noise , z + noise } ) );
        points.emplace_back( testXf( Vector3f{ cosf( angle ) * radius2 - noise , sinf( angle ) * radius2 + noise ,  1.0f - z - noise } ) );
    }

    //////////////////
    // General test //
    //////////////////

    Cone3<float> resultCone;
    auto fit = Cone3Approximation<float>();

    Cone3ApproximationParams approxiamtorParams;
    approxiamtorParams.coneFitterType = ConeFitterType::ApproximationPCM;

    [[maybe_unused]] auto error = fit.solve( points, resultCone, approxiamtorParams );

    //for debug in case of test fail.
    //std::cout << "Cone apex: " << resultCone.center() << " direction:" << resultCone.direction() << " heigh:"
    //  << resultCone.height << " angle:" << resultCone.angle * 180.0f / PI_F << " (degree)" << " error:" << error << std::endl;

    EXPECT_NEAR( resultCone.angle, coneAngle, 0.1f );
    EXPECT_NEAR( resultCone.height, coneHeight, 0.1f );
    EXPECT_LE( ( resultCone.apex() - coneApex ).length(), 0.1f );
    EXPECT_GT( dot( direction, resultCone.direction() ), 0.9f );

    //////////////////////////
    //    Hemisphere test   //
    //////////////////////////

    approxiamtorParams.coneFitterType = ConeFitterType::HemisphereSearchFit;

    error = fit.solve( points, resultCone, approxiamtorParams );

    //for debug in case of test fail.
    //std::cout << "Cone apex (Hem): " << resultCone.center() << " direction:" << resultCone.direction() << " heigh:"
    // << resultCone.height << " angle:" << resultCone.angle * 180.0f / PI_F << " (degree)" << " error:" << error << std::endl;

    EXPECT_NEAR( resultCone.angle, coneAngle, 0.1f );
    EXPECT_NEAR( resultCone.height, coneHeight, 0.1f );
    EXPECT_LE( ( resultCone.apex() - coneApex ).length(), 0.1f );
    EXPECT_GT( dot( direction, resultCone.direction() ), 0.9f );

    //////////////////////////
    // Use cone params test //
    //////////////////////////

    Cone3<float> noicedCone;
    Vector3f noiceVector = { 0.3234f , -0.2341f, 0.1234f };
    noicedCone.direction() = ( direction + noiceVector ).normalized();
    approxiamtorParams.coneFitterType = ConeFitterType::SpecificAxisFit;
    error = fit.solve( points, noicedCone, approxiamtorParams );

    //for debug in case of test fail.
    //std::cout << "Noiced cone apex: " << noicedCone.center() << " direction:" << noicedCone.direction() << " heigh:"
    //  << noicedCone.height << " angle:" << noicedCone.angle * 180.0f / PI_F << " (degree)" << " error:" << error << std::endl;

    EXPECT_NEAR( noicedCone.angle, coneAngle, 0.1f );
    EXPECT_NEAR( noicedCone.height, coneHeight, 0.1f );
    EXPECT_LE( ( noicedCone.apex() - coneApex ).length(), 0.1f );
    EXPECT_GT( dot( direction, noicedCone.direction() ), 0.9f );
}

} //namespace MR
