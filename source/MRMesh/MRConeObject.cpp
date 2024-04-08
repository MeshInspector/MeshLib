#include "MRConeObject.h"

#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRConeApproximator.h"
#include "MRMeshFwd.h"
#include "MRLine.h"
#include "MRGTest.h"

#include <iostream>
#include "MRArrow.h"

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


Matrix3f getConeRotationMatrix( const Vector3f& normal )
{
    return Matrix3f::rotation( Vector3f::plusZ(), normal );
}

}

MR_ADD_CLASS_FACTORY( ConeObject )

Vector3f ConeObject::getDirection( ViewportId id /*= {}*/ ) const
{
    return ( xf( id ).A * Vector3f::plusZ() ).normalized();
}

Vector3f ConeObject::getCenter( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

float ConeObject::getHeight( ViewportId id /*= {}*/ ) const
{
    return s_.get( id ).z.z;
}
void ConeObject::setHeight( float height, ViewportId id /*= {}*/ )
{
    auto direction = getDirection( id );
    auto currentXf = xf( id );
    auto radius = getNormalizedRadius_( id );
    currentXf.A = getConeRotationMatrix( direction ) * Matrix3f::scale( radius * height, radius * height, height );
    setXf( currentXf, id );
}


float ConeObject::getNormalizedRadius_( ViewportId id /*= {}*/ ) const
{
    return getBaseRadius( id ) / getHeight( id );
}
float ConeObject::getAngle( ViewportId id /*= {}*/ ) const
{
    return getAngleByNormalizedRadius( getNormalizedRadius_( id ) );
}

void ConeObject::setAngle( float angle, ViewportId id /*= {}*/ )
{
    setBaseRadius( getNormalizedRadiusByAngle( angle ) * getHeight( id ) );
}

void ConeObject::setDirection( const Vector3f& normal, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );

    currentXf.A = getConeRotationMatrix( normal ) * s_.get( id );
    setXf( currentXf, id );
}

void ConeObject::setCenter( const Vector3f& center, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.b = center;
    setXf( currentXf, id );
}

float ConeObject::getBaseRadius( ViewportId id /*= {}*/ ) const
{
    return s_.get( id ).x.x;
}

void ConeObject::setBaseRadius( float radius, ViewportId id /*= {}*/ )
{
    auto direction = getDirection( id );
    auto currentXf = xf( id );
    auto length = getHeight( id );
    currentXf.A = getConeRotationMatrix( direction ) * Matrix3f::scale( radius, radius, length );
    setXf( currentXf, id );
}

Vector3f ConeObject::getBasePoint( ViewportId id /*= {} */ ) const
{
    return getCenter( id ) + getDirection( id ) * getHeight( id );
}

ConeObject::ConeObject()
    : AddVisualProperties( 2 )
{}

ConeObject::ConeObject( const std::vector<Vector3f>& pointsToApprox )
    : ConeObject()
{
    // calculate cone parameters.
    Cone3<float> resultPCM, result;
    auto fit = Cone3Approximation<float>();

    Cone3ApproximationParams approxiamtorParams;
    approxiamtorParams.coneFitterType = ConeFitterType::HemisphereSearchFit;
    auto errorHemisphere = fit.solve( pointsToApprox, result, approxiamtorParams );

    approxiamtorParams.coneFitterType = ConeFitterType::ApproximationPCM;
    auto errorPCM = fit.solve( pointsToApprox, resultPCM, approxiamtorParams );

    if ( errorPCM < errorHemisphere )
        result = resultPCM;

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
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

const std::vector<FeatureObjectSharedProperty>& ConeObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Angle",     FeaturePropertyKind::angle,           &ConeObject::getAngle , &ConeObject::setAngle},
       {"Height",    FeaturePropertyKind::linearDimension, &ConeObject::getHeight, &ConeObject::setHeight},
       {"Center",    FeaturePropertyKind::position,        &ConeObject::getCenter, &ConeObject::setCenter},
       {"Main axis", FeaturePropertyKind::direction,       &ConeObject::getDirection, &ConeObject::setDirection},
    };
    return ret;
}

FeatureObjectProjectPointResult ConeObject::projectPoint( const Vector3f& point, ViewportId id /*= {}*/ ) const
{
    // Get direction, center, and angle of the cone from the specified viewport
    const Vector3f& n = getDirection( id );
    const Vector3f& center = getCenter( id );
    const float coneAngle = getAngle( id );

    // Calculate vector X as the difference between the point and the center of the cone
    Vector3f X = point - center;

    // Calculate the angle between vectors n (cone main axis)  and X (center normalyzed source point)
    float angleX = angle( n, X );

    // This means the projection will fall on the apex of the cone.
    if ( coneAngle + PI_F / 2.0 < angleX )
        return { center, -n };

    // Project vector X onto the cone main axis
    Vector3f K = n * MR::dot( X, n );
    Vector3f XK = ( X - K );

    // We find the point of intersection of the vector XK with the surface of the cone
    // and find a guide ventor along the surface of the cone to the projection point
    Vector3f D = K + XK.normalized() * ( K.length() * std::tan( coneAngle ) );
    Vector3f normD = D.normalized();

    // Calculate the projected point on the conical surface
    Vector3f projection = normD * dot( normD, X );

    // Calculate the normal at the projected point
    Vector3f Z = cross( n, normD );
    Vector3f normal = cross( Z, normD ).normalized();

    // Return the projection point and the normal
    return { projection + center, normal };
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



}
