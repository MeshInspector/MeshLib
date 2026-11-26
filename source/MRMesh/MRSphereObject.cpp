#include "MRSphereObject.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRIRenderObject.h"
#include "MRBestFit.h"
#include "MRVector.h"
#include <MRPch/MRJson.h>

namespace MR
{

MR_ADD_CLASS_FACTORY( SphereObject )

float SphereObject::getRadius( ViewportId id /*= {}*/ ) const
{
    return s_.get( id ).x.x;
}

Vector3f SphereObject::getCenter( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

void SphereObject::setRadius( float radius, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.A = Matrix3f::scale( radius );
    setXf( currentXf, id );
}

void SphereObject::setCenter( const Vector3f& center, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.b = center;
    setXf( currentXf, id );
}


const std::vector<FeatureObjectSharedProperty>& SphereObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Radius", FeaturePropertyKind::linearDimension, &SphereObject::getRadius, &SphereObject::setRadius},
       {"Center", FeaturePropertyKind::position,        &SphereObject::getCenter, &SphereObject::setCenter}
    };
    return ret;
}

FeatureObjectProjectPointResult SphereObject::projectPoint( const Vector3f& point, ViewportId id /*= {}*/ ) const
{
    const Vector3f& center = getCenter( id );
    const float radius = getRadius( id );

    auto X = point - center;
    auto normal = X.normalized();

    auto projection = center + normal * radius;
    return { projection, normal };
}

SphereObject::SphereObject()
    : AddVisualProperties( 2 )
{}

SphereObject::SphereObject( const std::vector<Vector3f>& pointsToApprox )
    : SphereObject()
{
    if ( const auto sphere = findBestSphere( Vector<Vector3d>( pointsToApprox.begin(), pointsToApprox.end() ) ) )
    {
        setCenter( Vector3f( sphere->center ) );
        setRadius( float( sphere->radius ) );
    }
}

std::shared_ptr<Object> SphereObject::shallowClone() const
{
    return std::make_shared<SphereObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> SphereObject::clone() const
{
    return std::make_shared<SphereObject>( ProtectedStruct{}, *this );
}

void SphereObject::swapBase_( Object& other )
{
    if ( auto sphereObject = other.asType<SphereObject>() )
        std::swap( *this, *sphereObject );
    else
        assert( false );
}

void SphereObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( SphereObject::StaticTypeName() );
}

void SphereObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

}
