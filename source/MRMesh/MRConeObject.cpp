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
constexpr int cDetailLevel = 64;
constexpr float thicknessArrow = 0.01f;
constexpr float cBaseRadius = 1.0f;
constexpr float cBaseHeight = 1.0f;

constexpr Vector3f base = Vector3f::plusZ();
constexpr Vector3f apex = { 0,0,0 };



float getFeatureRadiusByAngle( float angle )
{
    return cBaseHeight * std::tan( angle );
}
float getAngleByFeatureRadius( float fRadius )
{
    return std::atan( fRadius / cBaseHeight );
}


Matrix3f getRotationMatrix( const Vector3f& normal )
{
    return Matrix3f::rotation( Vector3f::plusZ(), normal );
}


std::shared_ptr<Mesh> makeFeatureCone( int resolution = cDetailLevel )
{

    auto mesh = std::make_shared<Mesh>( makeArrow( base, apex, thicknessArrow, cBaseRadius, cBaseHeight, resolution ) );

    return mesh;
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
    return xf().A.toScale().z;
}
void ConeObject::setHeight( float height )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto radius = getNormalyzedFeatueRadius();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( radius * height, radius * height, height );
    setXf( currentXf );
}


float ConeObject::getNormalyzedFeatueRadius( void ) const
{
    return ( xf().A.toScale().x + xf().A.toScale().y ) / 2.0f / getHeight();
}
float ConeObject::getAngle() const
{
    return getAngleByFeatureRadius( getNormalyzedFeatueRadius() );
}

void ConeObject::setAngle( float angle )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto featureRedius = getFeatureRadiusByAngle( angle );
    auto length = getHeight();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( featureRedius * length, featureRedius * length, length );
    setXf( currentXf );
}

void ConeObject::setDirection( const Vector3f& normal )
{
    auto currentXf = xf();
    currentXf.A = getRotationMatrix( normal ) * Matrix3f::scale( currentXf.A.toScale() );
    setXf( currentXf );
}

void ConeObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

ConeObject::ConeObject()
{
    constructMesh_();
}

ConeObject::ConeObject( const std::vector<Vector3f>& pointsToApprox )
{
    // create mesh
    constructMesh_();

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
    auto res = std::make_shared<ConeObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

std::shared_ptr<Object> ConeObject::clone() const
{
    auto res = std::make_shared<ConeObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
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
    ObjectMeshHolder::serializeFields_( root );
    root["Type"].append( ConeObject::TypeName() );
}

void ConeObject::constructMesh_()
{
    mesh_ = makeFeatureCone();
    setFlatShading( false );
    selectFaces( {} );
    selectEdges( {} );
    setDirtyFlags( DIRTY_ALL );
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
    auto radius = getFeatureRadiusByAngle( coneAngle );

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
    noicedCone.direction() = (direction + noiceVector).normalized();
    fit.solve( points, noicedCone, true );
    std::cout << "Noiced cone apex: " << noicedCone.center() << " direction:" << noicedCone.direction() << " heigh:" << noicedCone.height << " angle:" << noicedCone.angle * 180.0f / PI_F << " (degree)" << std::endl;

    EXPECT_NEAR( noicedCone.angle, coneAngle, 0.1f );
    EXPECT_NEAR( noicedCone.height, coneHeight, 0.1f );
    EXPECT_LE( ( noicedCone.apex() - coneApex ).length(), 0.1f );
    EXPECT_GT( dot( direction, noicedCone.direction() ), 0.9f );
}



}
