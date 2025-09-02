#include "MRCylinderObject.h"
#include "MRMatrix3.h"
#include "MRCylinder.h"
#include "MRCylinder3.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRCylinderApproximator.h"
#include "MRMeshNormals.h"
#include "MRLine.h"
#include "MRGTest.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRJson.h"
#include <iostream>

namespace MR
{

namespace
{

constexpr int phiResolution = 180;
constexpr int thetaResolution = 180;


Matrix3f getCylRotationMatrix( const Vector3f& normal )
{
    return Matrix3f::rotation( Vector3f::plusZ(), normal );
}

} // namespace

MR_ADD_CLASS_FACTORY( CylinderObject )

float CylinderObject::getLength( ViewportId id /*= {}*/ ) const
{
    return s_.get( id ).z.z;
}

void CylinderObject::setLength( float length, ViewportId id /*= {}*/ )
{
    auto direction = getDirection( id );
    auto currentXf = xf( id );
    auto radius = getRadius( id );
    currentXf.A = ( getCylRotationMatrix( direction ) * Matrix3f::scale( radius, radius, length ) );
    setXf( currentXf, id );
}

Vector3f CylinderObject::getBasePoint( ViewportId id /*= {} */ ) const
{
    return getCenter( id ) - getDirection( id ) * getLength( id ) * 0.5f;
}

float CylinderObject::getRadius( ViewportId id /*= {}*/ ) const
{
    // it is bad idea to use statement like this ( x + y ) / 2.0f; it increases instability. radius is changing during length update.
    return  s_.get( id ).x.x;
}

void CylinderObject::setRadius( float radius, ViewportId id /*= {}*/ )
{
    auto direction = getDirection( id );
    auto currentXf = xf( id );
    currentXf.A = getCylRotationMatrix( direction ) * Matrix3f::scale( radius, radius, getLength( id ) );
    setXf( currentXf, id );
}

Vector3f CylinderObject::getDirection( ViewportId id /*= {}*/ ) const
{
    return ( r_.get( id ) * Vector3f::plusZ() ).normalized();
}

Vector3f CylinderObject::getCenter( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

void CylinderObject::setDirection( const Vector3f& normal, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.A = getCylRotationMatrix( normal ) * s_.get( id );
    setXf( currentXf, id );
}

void CylinderObject::setCenter( const Vector3f& center, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.b = center;
    setXf( currentXf, id );
}

CylinderObject::CylinderObject()
    : AddVisualProperties( 2 )
{}

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
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

const std::vector<FeatureObjectSharedProperty>& CylinderObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Radius",    FeaturePropertyKind::linearDimension, &CylinderObject::getRadius, &CylinderObject::setRadius},
       {"Length",    FeaturePropertyKind::linearDimension, &CylinderObject::getLength, &CylinderObject::setLength},
       {"Center",    FeaturePropertyKind::position,        &CylinderObject::getCenter, &CylinderObject::setCenter},
       {"Main axis", FeaturePropertyKind::direction,       &CylinderObject::getDirection, &CylinderObject::setDirection},
    };
    return ret;
}

FeatureObjectProjectPointResult CylinderObject::projectPoint( const Vector3f& point, ViewportId id /*= {}*/ ) const
{
    // first, calculate the vector from the center of the cylinder to the projected point
    const auto& center = getCenter( id );
    const auto& direction = getDirection( id );
    auto radius = getRadius( id );

    auto X = point - center;

    float projectionLength = dot( X, direction );
    Vector3f K = direction * projectionLength;
    Vector3f normal = ( X - K ).normalized();
    Vector3f projection = K + normal * radius;

    return FeatureObjectProjectPointResult{ projection + center, normal };
}

} // namespace MR
