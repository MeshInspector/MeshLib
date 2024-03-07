#include "MRConeObject.h"

#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRMesh/MRDefaultFeatureObjectParams.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRConeApproximator.h"
#include "MRMeshFwd.h"
#include "MRLine.h"
#include "MRGTest.h"

#include <iostream>
#include "MRArrow.h"
#include "MRMatrix3Decompose.h"

namespace MR
{

namespace
{
constexpr float cBaseHeight = 1.0f;

float getNormalizedRadiusByAngle( float angle )
{
    return cBaseHeight * std::tan( angle );
}
float getAngleByNormalizedRadius( float fRadius )
{
    return std::atan( fRadius / cBaseHeight );
}


Matrix3f getRotationMatrix( const Vector3f& normal )
{
    return Matrix3f::rotation( Vector3f::plusZ(), normal );
}

}

MR_ADD_CLASS_FACTORY( ConeObject )

Vector3f ConeObject::getDirection() const
{
    return ( xf().A * Vector3f::plusZ() ).normalized();
}

Vector3f ConeObject::getCenter() const
{
    return xf().b;
}

float ConeObject::getHeight() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return s.z.z;
}
void ConeObject::setHeight( float height )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto radius = getNormalizedRadius_();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( radius * height, radius * height, height );
    setXf( currentXf );
}


float ConeObject::getNormalizedRadius_( void ) const
{
    return getBaseRadius() / getHeight();
}
float ConeObject::getAngle() const
{
    return getAngleByNormalizedRadius( getNormalizedRadius_() );
}

void ConeObject::setAngle( float angle )
{
    setBaseRadius( getNormalizedRadiusByAngle( angle ) * getHeight() );
}

void ConeObject::setDirection( const Vector3f& normal )
{
    auto currentXf = xf();
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    currentXf.A = getRotationMatrix( normal ) * s;
    setXf( currentXf );
}

void ConeObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

float ConeObject::getBaseRadius() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return s.x.x;
}

void ConeObject::setBaseRadius( float radius )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto length = getHeight();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( radius, radius, length );
    setXf( currentXf );
}

ConeObject::ConeObject()
{
    setDefaultFeatureObjectParams( *this );
}

ConeObject::ConeObject( const std::vector<Vector3f>& pointsToApprox )
    : ConeObject()
{
    // calculate cone parameters.
    Cone3<float> result;
    auto fit = Cone3Approximation<float>();
    fit.solve( pointsToApprox, result );

    // setup parameters
    setDirection( result.direction() );
    setCenter( result.center() );
    setAngle( result.angle );
    setHeight( result.height );
}

std::shared_ptr<Object> ConeObject::shallowClone() const
{
    return std::make_shared<ConeObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> ConeObject::clone() const
{
    return std::make_shared<ConeObject>( ProtectedStruct{}, *this );
}

void ConeObject::swapBase_( Object& other )
{
    if ( auto coneObject = other.asType<ConeObject>() )
        std::swap( *this, *coneObject );
    else
        assert( false );
}

void ConeObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( ConeObject::TypeName() );
}

void ConeObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype(*this)>( *this );
}

const std::vector<FeatureObjectSharedProperty>& ConeObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Angle",  &ConeObject::getAngle , &ConeObject::setAngle},
       {"Height", &ConeObject::getHeight, &ConeObject::setHeight},
       {"Center", &ConeObject::getCenter, &ConeObject::setCenter},
       {"Main axis", &ConeObject::getDirection, &ConeObject::setDirection},
    };
    return ret;
}

MeasurementPropertyParameters<RadiusVisualizePropertyType> ConeObject::getMeasurementParametersFor_( RadiusVisualizePropertyType index ) const
{
    (void)index; // Only one measurement.
    return {
        .center = Vector3f( 0, 0, 1 ),
        .vis = {
            .drawAsDiameter = true,
        },
    };
}

MeasurementPropertyParameters<AngleVisualizePropertyType> ConeObject::getMeasurementParametersFor_( AngleVisualizePropertyType index ) const
{
    (void)index; // Only one measurement.
    return {
        .rays = {
            Vector3f( 0.5f, 0, 0.5f ),
            Vector3f( -0.5f, 0, 0.5f ),
        },
        .vis = {
            .isConical = true,
        },
    };
}

MeasurementPropertyParameters<LengthVisualizePropertyType> ConeObject::getMeasurementParametersFor_( LengthVisualizePropertyType index ) const
{
    (void)index; // Only one measurement.
    return {
        .points = {
            Vector3f{},
            Vector3f( 0, 0, 1 ),
        },
    };
}

//////////////////
///// TESTS //////
//////////////////

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
    fit.solve( points, resultCone );
    std::cout << "Cone apex: " << resultCone.center() << " direction:" << resultCone.direction() << " heigh:" << resultCone.height << " angle:" << resultCone.angle * 180.0f / PI_F << " (degree)" << std::endl;

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
    fit.solve( points, noicedCone, true );
    std::cout << "Noiced cone apex: " << noicedCone.center() << " direction:" << noicedCone.direction() << " heigh:" << noicedCone.height << " angle:" << noicedCone.angle * 180.0f / PI_F << " (degree)" << std::endl;

    EXPECT_NEAR( noicedCone.angle, coneAngle, 0.1f );
    EXPECT_NEAR( noicedCone.height, coneHeight, 0.1f );
    EXPECT_LE( ( noicedCone.apex() - coneApex ).length(), 0.1f );
    EXPECT_GT( dot( direction, noicedCone.direction() ), 0.9f );
}



}
