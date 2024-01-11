#include "MRConeObject.h"

#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include <Eigen/Dense>
#include "MRConeApproximator.h"
#include "MRMeshFwd.h"
#include "MRLine.h"
#include "MRGTest.h"

#include <iostream>
#include "MRMeshNormals.h"
#include "MRMeshSubdivide.h"
#include "MRArrow.h"

namespace MR
{

namespace
{
constexpr int cDetailLevel = 64;
constexpr float thicknessArrow = 0.01f;
constexpr float cBaseRadius = 1.0f;
constexpr float cBaseHeight = 1.0f;

constexpr MR::Vector3f base = MR::Vector3f::plusZ();
constexpr MR::Vector3f apex = { 0,0,0 };



float getFeatureRediusByAngle( float angle )
{
    return cBaseHeight * std::tan( angle );
}
float getAngleByFeatureRedius( float fRadius )
{
    return std::atan( fRadius / cBaseHeight );
}


MR::Matrix3f getRotationMatrix( const Vector3f& normal )
{
    return Matrix3f::rotation( Vector3f::plusZ(), normal );
}


std::shared_ptr<MR::Mesh> makeFeatureCone( int resolution = cDetailLevel )
{

    auto mesh = std::make_shared<MR::Mesh>( makeArrow( base, apex, thicknessArrow, cBaseRadius, cBaseHeight, resolution ) );

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

float ConeObject::getLength() const
{
    return xf().A.toScale().z;
}
void ConeObject::setLength( float length )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto radius = getNormalyzedFeatueRadius();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( radius * length, radius * length, length );
    setXf( currentXf );
}


float ConeObject::getNormalyzedFeatueRadius( void ) const
{
    return ( xf().A.toScale().x + xf().A.toScale().y ) / 2.0f / getLength();
}
float ConeObject::getAngle() const
{
    return getAngleByFeatureRedius( getNormalyzedFeatueRadius() );
}

void ConeObject::setAngle( float angle )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto featureRedius = getFeatureRediusByAngle( angle );
    auto length = getLength();
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
    MR::Cone3<float> result;
    auto fit = Cone3Approximation<float>();
    fit.solve( pointsToApprox, result );

    // setup parameters
    setDirection( result.direction() );
    setCenter( result.center() );
    setAngle( result.angle );
    setLength( result.height );

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

}