#include "MRMesh/MRDefaultFeatureObjectParams.h"
#include "MRPch/MRSpdlog.h"
#include "MRCylinderObject.h"
#include "MRMatrix3.h"
#include "MRCylinder.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRMatrix3Decompose.h"
#include "MRCylinderApproximator.h"
#include "MRMeshFwd.h"
#include "MRLine.h"
#include "MRGTest.h"

#include <iostream>
#include "MRMeshNormals.h"

namespace MR
{

namespace
{

constexpr int phiResolution = 180;
constexpr int thetaResolution = 180;


Matrix3f getRotationMatrix( const Vector3f& normal )
{
    return Matrix3f::rotation( Vector3f::plusZ(), normal );
}

} // namespace

MR_ADD_CLASS_FACTORY( CylinderObject )

float CylinderObject::getLength() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return s.z.z;
}

void CylinderObject::setLength( float length )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto radius = getRadius();
    currentXf.A = ( getRotationMatrix( direction ) * Matrix3f::scale( radius, radius, length ) );
    setXf( currentXf );
}

float CylinderObject::getRadius() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    // it is bad idea to use statement like this ( x + y ) / 2.0f; it increases instability. radius is changing during length update.
    return  s.x.x;
}

void CylinderObject::setRadius( float radius )
{
    auto direction = getDirection();
    auto currentXf = xf();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( radius, radius, getLength() );
    setXf( currentXf );
}

Vector3f CylinderObject::getDirection() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return ( r * Vector3f::plusZ() ).normalized();
}

Vector3f CylinderObject::getCenter() const
{
    return xf().b;
}

void CylinderObject::setDirection( const Vector3f& normal )
{
    auto currentXf = xf();
    Matrix3f r, s;
    decomposeMatrix3( currentXf.A, r, s );
    currentXf.A = getRotationMatrix( normal ) * s;
    setXf( currentXf );
}

void CylinderObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

CylinderObject::CylinderObject()
{
    setDefaultFeatureObjectParams( *this );
}

CylinderObject::CylinderObject( const std::vector<Vector3f>& pointsToApprox )
    : CylinderObject()
{
    // calculate cylinder parameters.
    Cylinder3<float> result;
    auto fit = Cylinder3Approximation<float>();
    auto approxResult = fit.solveGeneral( pointsToApprox, result, phiResolution, thetaResolution );
    if ( approxResult < 0 )
    {
        spdlog::warn( "CylinderObject :: unable to creater feature object cylinder." );
        return;
    }

    // setup parameters
    setRadius( result.radius );
    setLength( result.length );
    setDirection( result.direction() );
    setCenter( result.center() );
}

std::shared_ptr<Object> CylinderObject::shallowClone() const
{
    return std::make_shared<CylinderObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> CylinderObject::clone() const
{
    return std::make_shared<CylinderObject>( ProtectedStruct{}, *this );
}

void CylinderObject::swapBase_( Object& other )
{
    if ( auto cylinderObject = other.asType<CylinderObject>() )
        std::swap( *this, *cylinderObject );
    else
        assert( false );
}

void CylinderObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( CylinderObject::TypeName() );
}

void CylinderObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype(*this)>( *this );
}

MeasurementPropertyParameters<RadiusVisualizePropertyType> CylinderObject::getMeasurementParametersFor_( RadiusVisualizePropertyType index ) const
{
    (void)index; // Only one measurement.
    return {
        .vis = {
            .drawAsDiameter = true,
        },
    };
}

MeasurementPropertyParameters<LengthVisualizePropertyType> CylinderObject::getMeasurementParametersFor_( LengthVisualizePropertyType index ) const
{
    (void)index; // Only one measurement.
    return {
        .points = {
            Vector3f( 0, 0, -0.5f ),
            Vector3f( 0, 0, 0.5f ),
        },
    };
}

const std::vector<FeatureObjectSharedProperty>& CylinderObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Radius", &CylinderObject::getRadius, &CylinderObject::setRadius},
       {"Length", &CylinderObject::getLength, &CylinderObject::setLength},
       {"Center", &CylinderObject::getCenter, &CylinderObject::setCenter},
       {"Main axis", &CylinderObject::getDirection, &CylinderObject::setDirection},
    };
    return ret;
}


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
