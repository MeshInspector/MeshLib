#include "MRSphereObject.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRIRenderObject.h"
#include "MRSymMatrix4.h"
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
    // find best radius and center
    SymMatrix4d accumA;
    Vector4d accumB;
    for ( const auto& pta : pointsToApprox )
    {
        const Vector4d vec{ 2.0 * pta.x, 2.0 * pta.y, 2.0 * pta.z, -1.0 };
        accumA += outerSquare( vec );
        accumB += vec * Vector3d( pta ).lengthSq();
    }
    const Vector4d res = accumA.pseudoinverse() * accumB;
    setCenter( Vector3f{ float( res.x ), float( res.y ), float( res.z ) } );
    double rSq = res.x * res.x + res.y * res.y + res.z * res.z - res.w;
    assert( rSq >= 0.0 );
    setRadius( float( sqrt( std::max( rSq, 0.0 ) ) ) );
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
